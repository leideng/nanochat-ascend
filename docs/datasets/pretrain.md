# Pretrain Dataset

## Role In The Pipeline

The pretrain dataset is the raw language-modeling corpus used by `scripts/base_train.py`. It teaches the base model general language modeling before any chat fine-tuning begins.

## Source

The repository downloads the pretraining corpus from the Hugging Face dataset repo:

- `karpathy/fineweb-edu-100b-shuffle`

The local target directory is derived from `configs/global.yaml` and resolves to:

```text
.cache/dataset/pretrain/fineweb-edu-100b-shuffle
```

## On-Disk Format

The pretraining dataset is stored as parquet files. The code expects each row to contain a `text` column.

Typical row shape:

```json
{
  "text": "Shipment & Transport-Sea, Air, Rail, Road, Pipeline..."
}
```

## How It Is Loaded

The main loading utilities live in:

- `nanochat/dataset.py`
- `nanochat/dataloader.py`

Key behaviors:

- all `.parquet` files in the configured directory are enumerated
- the last parquet file is reserved as the validation split
- all earlier parquet files form the training split
- row groups are read in batches for efficiency

The relevant split behavior in `nanochat.dataset.parquets_iter_batched` is:

- `train`: all parquet files except the last
- `val`: only the last parquet file

## How Text Becomes Training Batches

For pretraining, raw documents are tokenized and packed by the distributed data loader using a BOS-aligned best-fit strategy.

That gives the repo several properties:

- every training row begins with a BOS token
- examples are packed into fixed-length training rows
- the loader is DDP-aware
- tokenized text is emitted directly as autoregressive `inputs` and `targets`

This data path feeds:

- `scripts/base_train.py`
- the BPB path inside `scripts/base_eval.py`

## Why This Dataset Is Separate

This corpus is meant for base language modeling, not instruction tuning. It does not contain structured roles such as `user`, `assistant`, or `system`.

That distinction matters:

- pretraining learns broad language statistics
- SFT and RL later teach chat behavior and task-specific response structure

## Operational Notes

- The dataset is downloaded by `python -m nanochat.dataset`, which is wrapped by `bash runs/prepare_dataset.sh`.
- The repo expects the parquet directory to exist before base training starts.
- Because the last parquet file is used for validation, users should preserve the expected file set when mirroring or syncing the corpus.
