import os

import pytest
import torch

from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.tokenizer import get_tokenizer

from tests.conftest import has_npu


pytestmark = [
    pytest.mark.npu,
    pytest.mark.skipif(
        not has_npu() or os.environ.get("NANOCHAT_RUN_NPU_TESTS") != "1",
        reason="NPU tests require Ascend hardware and NANOCHAT_RUN_NPU_TESTS=1",
    ),
]


def test_dataloader_npu_medium_batch():
    tokenizer = get_tokenizer()
    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer,
        B=2,
        T=128,
        split="train",
        device=torch.device("npu"),
        tokenizer_batch_size=64,
        buffer_size=64,
    )

    inputs, targets, state = next(loader)

    assert inputs.device.type == "npu"
    assert targets.device.type == "npu"
    assert inputs.shape == (2, 128)
    assert targets.shape == (2, 128)
    assert state["epoch"] >= 1
