"""
The base/pretraining dataset is a set of parquet files.
You should first download the dataset and saving into config.pretrain_dataset.
This file contains utilities for iterating over the parquet files and yielding documents from it.

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import shutil
import time
import pyarrow.parquet as pq
from nanochat.common import get_global_config
from datasets import load_dataset


def _progress(msg: str) -> None:
    """Print progress message and flush so it appears immediately."""
    print(msg, flush=True)

def list_parquet_files(data_dir=None):
    """ Looks into a data dir and returns full paths to all parquet files. """
    data_dir = get_global_config().pretrain_dataset if data_dir is None else data_dir
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

def parquets_iter_batched(split, data_dir=None, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files(data_dir) if data_dir is not None else list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts


def download_url_datasets():
    """
    Download the datasets from the URL and save them to the local cache directory.

    sft_dataset: .cache/dataset/sft/identity_conversations.jsonl #source: https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
    simple_spelling_dataset: .cache/dataset/sft/words_alpha.txt  #source:https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt
    """
    #just curl down the files and save them to the local cache directory.
    sft_dataset_url = "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl"
    simple_spelling_dataset_url = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"
    sft_dataset_path = get_global_config().sft_dataset
    simple_spelling_dataset_path = get_global_config().simple_spelling_dataset
    os.makedirs(os.path.dirname(sft_dataset_path), exist_ok=True)
    os.makedirs(os.path.dirname(simple_spelling_dataset_path), exist_ok=True)
    os.system(f"curl -o {sft_dataset_path} {sft_dataset_url}")
    os.system(f"curl -o {simple_spelling_dataset_path} {simple_spelling_dataset_url}")


def _ensure_dataset(step: int, total: int, name: str, local_path: str, hub_id: str):
    """Load dataset from local path or download from Hub, with progress prints. Returns the dataset."""
    _progress(f"[{step}/{total}] {name}: checking {local_path} ...")
    t0 = time.monotonic()
    try:
        ds = load_dataset(local_path, split="train")
        elapsed = time.monotonic() - t0
        _progress(f"[{step}/{total}] {name}: loaded locally in {elapsed:.1f}s")
        return ds
    except Exception:
        if os.path.exists(local_path):
            _progress(f"[{step}/{total}] {name}: removing incomplete cache at {local_path} ...")
            shutil.rmtree(local_path)
        _progress(f"[{step}/{total}] {name}: downloading from Hub '{hub_id}' (may show progress below) ...")
        t_dl = time.monotonic()
        ds = load_dataset(hub_id, split="train")
        _progress(f"[{step}/{total}] {name}: download finished in {time.monotonic() - t_dl:.1f}s")
        _progress(f"[{step}/{total}] {name}: saving to disk {local_path} ...")
        t_save = time.monotonic()
        ds.save_to_disk(local_path)
        _progress(f"[{step}/{total}] {name}: saved in {time.monotonic() - t_save:.1f}s (total {time.monotonic() - t0:.1f}s)")
        return ds


def download_huggingface_datasets():
    """
    Download the datasets from the Hugging Face Hub and save them to the local cache directory.

    pretrain_dataset: .cache/dataset/pretrain/fineweb-edu-100b-shuffle-sample #source: https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle
    eval_dataset: .cache/dataset/eval  #source: https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
    allenai_arc_dataset: .cache/dataset/ai2_arc #source https://huggingface.co/datasets/allenai/ai2_arc
    openai_gsm8k_dataset: .cache/dataset/gsm8k #source: https://huggingface.co/datasets/openai/gsm8k
    openai_humaneval_dataset: .cache/dataset/humaneval #source: https://huggingface.co/datasets/openai/openai_humaneval
    cais_mmlu_dataset: .cache/dataset/mmlu #source: https://huggingface.co/datasets/cais/mmlu
    huggingface_tb_smol_smoltalk_dataset: .cache/dataset/smol-smoltalk #source: https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
    """
    cfg = get_global_config()
    pretrain_dataset_path = cfg.pretrain_dataset
    eval_dataset_path = cfg.eval_dataset
    allenai_arc_dataset_path = cfg.allenai_arc_dataset
    openai_gsm8k_dataset_path = cfg.openai_gsm8k_dataset
    openai_humaneval_dataset_path = cfg.openai_humaneval_dataset
    cais_mmlu_dataset_path = cfg.cais_mmlu_dataset
    huggingface_tb_smol_smoltalk_dataset_path = cfg.huggingface_tb_smol_smoltalk_dataset

    total = 7
    _progress(f"Preparing {total} Hugging Face datasets (Hub downloads may show progress bars below).")

    pretrain_dataset = _ensure_dataset(
        1, total, "pretrain", pretrain_dataset_path, "karpathy/fineweb-edu-100b-shuffle"
    )
    eval_dataset = _ensure_dataset(2, total, "eval", eval_dataset_path, "karpathy/eval_bundle")
    allenai_arc_dataset = _ensure_dataset(
        3, total, "allenai/ai2_arc", allenai_arc_dataset_path, "allenai/ai2_arc"
    )
    openai_gsm8k_dataset = _ensure_dataset(
        4, total, "openai/gsm8k", openai_gsm8k_dataset_path, "openai/gsm8k"
    )
    openai_humaneval_dataset = _ensure_dataset(
        5, total, "openai_humaneval", openai_humaneval_dataset_path, "openai/openai_humaneval"
    )
    cais_mmlu_dataset = _ensure_dataset(
        6, total, "cais/mmlu", cais_mmlu_dataset_path, "cais/mmlu"
    )
    huggingface_tb_smol_smoltalk_dataset = _ensure_dataset(
        7, total, "smol-smoltalk", huggingface_tb_smol_smoltalk_dataset_path, "HuggingFaceTB/smol-smoltalk"
    )

    _progress("All Hugging Face datasets ready.")    

# you can run it in the project root directory by running `python -m nanochat.dataset`
if __name__ == "__main__": 
    print("Downloading URL datasets...")   
    download_url_datasets()    

    print("Downloading Hugging Face datasets...")
    download_huggingface_datasets()

    parquet_files = list_parquet_files()
    
    print("="*20 + "Parquet files info" + "="*20)
    print(f"List of parquet files: {parquet_files}")
    print(f"Number of parquet files: {len(parquet_files)}")
    
    pf_first = pq.ParquetFile(parquet_files[0])
    pf_last = pq.ParquetFile(parquet_files[-1])

    print("="*20 + "First parquet file metadata" + "="*20)
    print(f"Number of row groups in the first parquet file: {pf_first.metadata.num_row_groups}")
    print(f"Number of rows in the first parquet file: {pf_first.metadata.num_rows}")
    print(f"Number of columns in the first parquet file: {pf_first.metadata.num_columns}")

    print("="*20 + "Last parquet file metadata" + "="*20)
    print(f"Number of rows in the last parquet file: {pf_last.metadata.num_rows}")
    print(f"Number of row groups in the last parquet file: {pf_last.metadata.num_row_groups}")
    print(f"Number of columns in the last parquet file: {pf_last.metadata.num_columns}")

    print("="*20 + "Iterating through the dataset..." + "="*20)
    idx = 0
    for texts in parquets_iter_batched(split="train"):
        n_samples = len(texts)
        first_sample = texts[0]
        # print the first sample in a short form with only start and end of the string
        # in addition, print it as a raw string by using repr(first_sample_short) 
        # such that special characters like \n are not escaped
        first_sample_short = first_sample[:30]+"..."+first_sample[-30:] if len(first_sample) > 30 else first_sample
        print(f"Batch {idx}: n_samples={n_samples}, first sample (short)={repr(first_sample_short)}")
        idx += 1
        