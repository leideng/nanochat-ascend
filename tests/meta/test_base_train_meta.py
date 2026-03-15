import pytest

from scripts import base_train


@pytest.mark.meta
def test_build_model_meta_uses_meta_device_and_rounded_dimensions():
    parser = base_train.build_arg_parser()
    args = parser.parse_args(
        [
            "--depth",
            "3",
            "--aspect-ratio",
            "10",
            "--head-dim",
            "16",
            "--max-seq-len",
            "32",
            "--window-pattern",
            "SL",
        ]
    )

    model = base_train.build_model_meta(args, vocab_size=128)

    assert next(model.parameters()).device.type == "meta"
    assert model.config.n_embd == 32
    assert model.config.n_head == 2
    assert model.window_sizes[-1] == (32, 0)
