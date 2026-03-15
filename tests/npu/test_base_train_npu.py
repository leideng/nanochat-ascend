import os

import pytest
import torch

from scripts import base_train

from tests.conftest import has_npu


pytestmark = [
    pytest.mark.npu,
    pytest.mark.skipif(
        not has_npu() or os.environ.get("NANOCHAT_RUN_NPU_TESTS") != "1",
        reason="NPU tests require Ascend hardware and NANOCHAT_RUN_NPU_TESTS=1",
    ),
]


def test_base_train_npu_medium_smoke():
    if "NANOCHAT_CONFIG" not in os.environ:
        pytest.skip("Run 'source runs/set_env.sh' before NPU smoke tests")

    base_train.main(
        [
            "--run",
            "dummy",
            "--device-type",
            "npu",
            "--depth",
            "4",
            "--aspect-ratio",
            "32",
            "--head-dim",
            "64",
            "--max-seq-len",
            "128",
            "--device-batch-size",
            "2",
            "--total-batch-size",
            "512",
            "--num-iterations",
            "2",
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
