import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from nanochat import report
from scripts import base_eval

from tests.conftest import DummyReport


class DummyEvalModel:
    def __init__(self, device):
        self._device = device

    def get_device(self):
        return self._device


def _bpb_loader():
    x = torch.ones((1, 8), dtype=torch.long)
    y = torch.ones((1, 8), dtype=torch.long)
    while True:
        yield x.clone(), y.clone()


@pytest.mark.cpu
def test_evaluate_core_reads_bundle_and_computes_centered_scores(tmp_path, monkeypatch, dummy_tokenizer):
    eval_dir = tmp_path / "eval_bundle"
    data_dir = eval_dir / "eval_data"
    data_dir.mkdir(parents=True)
    (eval_dir / "core.yaml").write_text(
        yaml.safe_dump(
            {
                "icl_tasks": [
                    {
                        "label": "ToyTask",
                        "icl_task_type": "multiple_choice",
                        "dataset_uri": "toy.jsonl",
                        "num_fewshot": [0],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with open(eval_dir / "eval_meta_data.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Eval Task", "Random baseline"])
        writer.writeheader()
        writer.writerow({"Eval Task": "ToyTask", "Random baseline": "25"})
    with open(data_dir / "toy.jsonl", "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"query": "q", "choices": ["a", "b"], "gold": 0}) + "\n")
        handle.write(json.dumps({"query": "q2", "choices": ["a", "b"], "gold": 1}) + "\n")

    monkeypatch.setattr(base_eval, "get_global_config", lambda: SimpleNamespace(eval_dataset=str(eval_dir)))
    monkeypatch.setattr(base_eval, "evaluate_task", lambda *args, **kwargs: 0.75)

    result = base_eval.evaluate_core(DummyEvalModel(torch.device("cpu")), dummy_tokenizer, torch.device("cpu"), max_per_task=1)

    assert result["results"]["ToyTask"] == 0.75
    assert result["centered_results"]["ToyTask"] == pytest.approx((0.75 - 0.25) / (1.0 - 0.25))
    assert result["core_metric"] == result["centered_results"]["ToyTask"]


@pytest.mark.cpu
def test_base_eval_main_runs_tiny_cpu_smoke(tmp_path, monkeypatch, dummy_tokenizer):
    report_sink = DummyReport()
    base_eval_dir = tmp_path / "base_eval"
    model = DummyEvalModel(torch.device("cpu"))

    monkeypatch.setattr(base_eval, "compute_init", lambda device_type: (False, 0, 0, 1, torch.device(device_type)))
    monkeypatch.setattr(base_eval, "load_model", lambda *args, **kwargs: (model, dummy_tokenizer, {"step": 7, "model_config": {"sequence_len": 8}}))
    monkeypatch.setattr(base_eval, "get_token_bytes", lambda device: torch.ones(dummy_tokenizer.get_vocab_size(), dtype=torch.int64, device=device))
    monkeypatch.setattr(base_eval, "tokenizing_distributed_data_loader_bos_bestfit", lambda *args, **kwargs: _bpb_loader())
    monkeypatch.setattr(base_eval, "evaluate_bpb", lambda *args, **kwargs: 1.234)
    monkeypatch.setattr(base_eval, "evaluate_core", lambda *args, **kwargs: {"results": {"ToyTask": 0.5}, "centered_results": {"ToyTask": 0.5}, "core_metric": 0.5})
    monkeypatch.setattr(base_eval, "get_global_config", lambda: SimpleNamespace(base_eval_dir=str(base_eval_dir)))
    monkeypatch.setattr(report, "get_report", lambda: report_sink)

    base_eval.main(
        [
            "--eval",
            "core,bpb",
            "--device-type",
            "cpu",
            "--device-batch-size",
            "1",
            "--split-tokens",
            "16",
            "--model-tag",
            "toy",
            "--step",
            "7",
        ]
    )

    output_csv = base_eval_dir / "base_model_000007.csv"
    assert output_csv.exists()
    assert "ToyTask" in output_csv.read_text(encoding="utf-8")
    assert report_sink.logged
