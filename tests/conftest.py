from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch


class DummyTokenizer:
    def __init__(self, vocab_size=64, bos_token_id=1):
        self._vocab_size = vocab_size
        self._bos_token_id = bos_token_id

    def get_vocab_size(self):
        return self._vocab_size

    def get_bos_token_id(self):
        return self._bos_token_id

    def encode_special(self, text):
        if text in {"<|bos|>", "<|endoftext|>"}:
            return self._bos_token_id
        raise KeyError(text)

    def encode(self, text, prepend=None, append=None, num_threads=None):
        del num_threads
        if isinstance(text, list):
            return [self.encode(item, prepend=prepend, append=append) for item in text]
        ids = [2 + (ord(ch) % (self._vocab_size - 2)) for ch in text]
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids = [prepend_id] + ids
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids = ids + [append_id]
        return ids

    __call__ = encode

    def decode(self, ids):
        return "".join(chr((idx - 2) % 26 + 97) for idx in ids if idx != self._bos_token_id)


class DummyReport:
    def __init__(self):
        self.logged = []

    def log(self, *args, **kwargs):
        self.logged.append((args, kwargs))


def write_parquet_file(path: Path, row_groups):
    schema = pa.schema([("text", pa.string())])
    with pq.ParquetWriter(path, schema) as writer:
        for texts in row_groups:
            table = pa.Table.from_pydict({"text": texts}, schema=schema)
            writer.write_table(table)


def make_global_config(tmp_path: Path):
    return SimpleNamespace(
        pretrain_dataset=str(tmp_path / "pretrain"),
        eval_dataset=str(tmp_path / "eval_bundle"),
        base_checkpoints_dir=str(tmp_path / "checkpoints"),
        base_eval_dir=str(tmp_path / "base_eval"),
        tokenizer_dir=str(tmp_path / "tokenizer"),
        report_dir=str(tmp_path / "report"),
        enforce_eager=True,
    )


def has_npu():
    return hasattr(torch, "npu") and torch.npu.is_available()


@pytest.fixture
def dummy_tokenizer():
    return DummyTokenizer()
