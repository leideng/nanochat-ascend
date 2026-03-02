"""
The base/pretraining dataset is a set of parquet files.
You should first download the dataset and saving into config.pretrain_dataset.
This file contains utilities for iterating over the parquet files and yielding documents from it.

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import pyarrow.parquet as pq
from nanochat.common import get_global_config

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


# you can run it in the project root directory by running `python -m nanochat.dataset`
if __name__ == "__main__":    
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

    print("="*20 + "First parquet file metadata" + "="*20)
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
        