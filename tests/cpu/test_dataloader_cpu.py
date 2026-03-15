import pytest
import torch

from nanochat import dataloader

from tests.conftest import write_parquet_file


@pytest.mark.cpu
def test_bos_bestfit_loader_emits_small_cpu_batch(tmp_path, monkeypatch, dummy_tokenizer):
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    write_parquet_file(train_path, [["alpha", "bb", "c"], ["dddd", "ee"]])
    write_parquet_file(val_path, [["validation"]])

    monkeypatch.setattr(dataloader, "get_dist_info", lambda: (False, 0, 0, 1))
    monkeypatch.setattr(dataloader, "list_parquet_files", lambda: [str(train_path), str(val_path)])

    loader = dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(
        dummy_tokenizer,
        B=2,
        T=5,
        split="train",
        device="cpu",
        tokenizer_batch_size=2,
        buffer_size=3,
    )

    inputs, targets, state = next(loader)

    assert inputs.shape == (2, 5)
    assert targets.shape == (2, 5)
    assert inputs.device.type == "cpu"
    assert torch.equal(inputs[:, 1:], targets[:, :-1])
    assert torch.all(inputs[:, 0] == dummy_tokenizer.get_bos_token_id())
    assert {"pq_idx", "rg_idx", "epoch"} <= state.keys()
