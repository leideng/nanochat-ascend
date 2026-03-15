from types import SimpleNamespace

import pytest
import torch

from nanochat import report
from scripts import base_train

from tests.conftest import DummyReport


def _train_loader():
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    y = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
    state = {"pq_idx": 0, "rg_idx": 0, "epoch": 1}
    while True:
        yield x.clone(), y.clone(), state.copy()


def _val_loader():
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    y = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
    while True:
        yield x.clone(), y.clone()


@pytest.mark.cpu
def test_base_train_main_runs_tiny_cpu_smoke(tmp_path, monkeypatch, dummy_tokenizer):
    saved = []
    report_sink = DummyReport()
    config = SimpleNamespace(base_checkpoints_dir=str(tmp_path / "checkpoints"), enforce_eager=True)

    monkeypatch.setattr(base_train, "print_banner", lambda: None)
    monkeypatch.setattr(base_train, "compute_init", lambda device_type: (False, 0, 0, 1, torch.device(device_type)))
    monkeypatch.setattr(base_train, "get_global_config", lambda: config)
    monkeypatch.setattr(base_train, "get_tokenizer", lambda: dummy_tokenizer)
    monkeypatch.setattr(base_train, "get_token_bytes", lambda device: torch.ones(dummy_tokenizer.get_vocab_size(), dtype=torch.int64, device=device))
    monkeypatch.setattr(base_train, "tokenizing_distributed_data_loader_with_state_bos_bestfit", lambda *args, **kwargs: _train_loader())
    monkeypatch.setattr(base_train, "tokenizing_distributed_data_loader_bos_bestfit", lambda *args, **kwargs: _val_loader())
    monkeypatch.setattr(base_train, "save_checkpoint", lambda *args, **kwargs: saved.append((args, kwargs)))
    monkeypatch.setattr(report, "get_report", lambda: report_sink)

    base_train.main(
        [
            "--run",
            "dummy",
            "--device-type",
            "cpu",
            "--depth",
            "2",
            "--aspect-ratio",
            "16",
            "--head-dim",
            "8",
            "--max-seq-len",
            "8",
            "--device-batch-size",
            "1",
            "--total-batch-size",
            "8",
            "--num-iterations",
            "1",
            "--eval-every",
            "-1",
            "--core-metric-every",
            "-1",
            "--sample-every",
            "-1",
            "--save-every",
            "-1",
        ]
    )

    assert len(saved) == 1
    assert saved[0][0][0].endswith("/d2")
    assert report_sink.logged
