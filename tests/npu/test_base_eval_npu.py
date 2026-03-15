import os

import pytest

from scripts import base_eval

from tests.conftest import has_npu


pytestmark = [
    pytest.mark.npu,
    pytest.mark.skipif(
        not has_npu() or os.environ.get("NANOCHAT_RUN_NPU_TESTS") != "1",
        reason="NPU tests require Ascend hardware and NANOCHAT_RUN_NPU_TESTS=1",
    ),
]


def test_base_eval_npu_medium_smoke():
    if "NANOCHAT_CONFIG" not in os.environ:
        pytest.skip("Set NANOCHAT_CONFIG to an Ascend-ready config before running NPU tests")

    model_tag = os.environ.get("NANOCHAT_TEST_MODEL_TAG")
    if not model_tag:
        pytest.skip("Set NANOCHAT_TEST_MODEL_TAG to the checkpoint tag to evaluate")

    base_eval.main(
        [
            "--eval",
            "bpb,core",
            "--device-type",
            "npu",
            "--device-batch-size",
            "4",
            "--split-tokens",
            str(4 * 128 * 4),
            "--max-per-task",
            "32",
            "--model-tag",
            model_tag,
        ]
    )
