"""
The base/pretraining dataset is a set of parquet files.
You should first download the dataset and saving into config.pretrain_dataset.
This file contains utilities for iterating over the parquet files and yielding documents from it.

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import shutil
import pyarrow.parquet as pq
from nanochat.common import get_global_config
from datasets import load_dataset

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

    pretrain_dataset_path = get_global_config().pretrain_dataset
    eval_dataset_path = get_global_config().eval_dataset
    allenai_arc_dataset_path = get_global_config().allenai_arc_dataset
    openai_gsm8k_dataset_path = get_global_config().openai_gsm8k_dataset
    openai_humaneval_dataset_path = get_global_config().openai_humaneval_dataset
    cais_mmlu_dataset_path = get_global_config().cais_mmlu_dataset
    huggingface_tb_smol_smoltalk_dataset_path = get_global_config().huggingface_tb_smol_smoltalk_dataset

    print(f"Downloading pretrain dataset to {pretrain_dataset_path}...")
    try:
        pretrain_dataset = load_dataset(pretrain_dataset_path, split="train")
        print(f"Pretrain dataset loaded successfully locally from {pretrain_dataset_path}")
    except Exception:
        print(f"Pretrain dataset not found locally at {pretrain_dataset_path}, downloading from Hub...")
        # Local cache corrupted or incomplete; remove and re-download from Hub
        if os.path.exists(pretrain_dataset_path):
            print(f"Removing local cache at {pretrain_dataset_path}...")
            shutil.rmtree(pretrain_dataset_path)
        print(f"Downloading pretrain dataset from Hugging Face Hub...")
        pretrain_dataset = load_dataset("karpathy/fineweb-edu-100b-shuffle", split="train")
        print(f"Saving pretrain dataset to {pretrain_dataset_path}...")
        pretrain_dataset.save_to_disk(pretrain_dataset_path)
        print(f"Pretrain dataset saved successfully to {pretrain_dataset_path}")

    print(f"Downloading eval dataset from {eval_dataset_path}...")
    try:
        eval_dataset = load_dataset(eval_dataset_path, split="train")
    except Exception:
        # Local cache corrupted or incomplete; remove and re-download from Hub
        if os.path.exists(eval_dataset_path):
            shutil.rmtree(eval_dataset_path)
        eval_dataset = load_dataset("karpathy/eval_bundle", split="train")
        eval_dataset.save_to_disk(eval_dataset_path)

    try:
        allenai_arc_dataset = load_dataset(allenai_arc_dataset_path, split="train")
    except Exception:
        # Local cache corrupted or incomplete; remove and re-download from Hub
        if os.path.exists(allenai_arc_dataset_path):
            shutil.rmtree(allenai_arc_dataset_path)
        allenai_arc_dataset = load_dataset("allenai/ai2_arc", split="train")
        allenai_arc_dataset.save_to_disk(allenai_arc_dataset_path)
        
    try:
        openai_gsm8k_dataset = load_dataset(openai_gsm8k_dataset_path, split="train")
    except Exception:
        # Local cache corrupted or incomplete; remove and re-download from Hub
        if os.path.exists(openai_gsm8k_dataset_path):
            shutil.rmtree(openai_gsm8k_dataset_path)
        openai_gsm8k_dataset = load_dataset("openai/gsm8k", split="train")
        openai_gsm8k_dataset.save_to_disk(openai_gsm8k_dataset_path)
        
    try:
        openai_humaneval_dataset = load_dataset(openai_humaneval_dataset_path, split="train")
    except Exception:
        # Local cache corrupted or incomplete; remove and re-download from Hub
        if os.path.exists(openai_humaneval_dataset_path):
            shutil.rmtree(openai_humaneval_dataset_path)
        openai_humaneval_dataset = load_dataset("openai/openai_humaneval", split="train")
        openai_humaneval_dataset.save_to_disk(openai_humaneval_dataset_path)
    try:
        cais_mmlu_dataset = load_dataset(cais_mmlu_dataset_path, split="train")
    except Exception:
        # Local cache corrupted or incomplete; remove and re-download from Hub
        if os.path.exists(cais_mmlu_dataset_path):
            shutil.rmtree(cais_mmlu_dataset_path)
        cais_mmlu_dataset = load_dataset("cais/mmlu", split="train")
        cais_mmlu_dataset.save_to_disk(cais_mmlu_dataset_path)

    try:
        huggingface_tb_smol_smoltalk_dataset = load_dataset(huggingface_tb_smol_smoltalk_dataset_path, split="train")
    except Exception:
        # Local cache corrupted or incomplete; remove and re-download from Hub
        if os.path.exists(huggingface_tb_smol_smoltalk_dataset_path):
            shutil.rmtree(huggingface_tb_smol_smoltalk_dataset_path)
        huggingface_tb_smol_smoltalk_dataset = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
        huggingface_tb_smol_smoltalk_dataset.save_to_disk(huggingface_tb_smol_smoltalk_dataset_path)    

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
        