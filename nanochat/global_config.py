"""
Configuration for the nanochat-ascend model.
"""

from dataclasses import dataclass, fields
import json
import os

import torch
import yaml


def _autodetect_config_device() -> str:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    return "cpu"

@dataclass(frozen=True)
class GlobalConfig:
    device: str = _autodetect_config_device()
    enforce_eager: bool = True

    ## dataset paths
    pretrain_dataset: str = ""
    identity_conversations_dataset: str = ""
    simple_spelling_dataset: str = ""
    eval_dataset: str = ""
    allenai_arc_dataset: str = ""
    openai_gsm8k_dataset: str = ""
    openai_humaneval_dataset: str = ""
    cais_mmlu_dataset: str = ""
    huggingface_tb_smol_smoltalk_dataset: str = ""

    ## output paths
    output_dir: str = ""
    base_checkpoints_dir: str = ""
    chatsft_checkpoints_dir: str = ""
    chatrl_checkpoints_dir: str = ""
    base_eval_dir: str = ""
    chatsft_eval_dir: str = ""
    chatrl_eval_dir: str = ""
    tokenizer_dir: str = ""
    report_dir: str = ""

    @staticmethod
    def _resolve_path(base: str, value: str) -> str:
        if not value:
            return value
        if os.path.isabs(value) or not base:
            return value
        return os.path.join(base, value)

    @classmethod
    def _expand_hierarchical_paths(cls, data: dict) -> dict:
        expanded = dict(data)
        expanded.pop("dataset", None)
        expanded.pop("checkpoint", None)
        expanded.pop("output", None)

        dataset_cfg = data.get("dataset")
        if dataset_cfg is not None:
            if not isinstance(dataset_cfg, dict):
                raise ValueError("Config key dataset must be a mapping")
            dataset_root = dataset_cfg.get("root", "")
            top_level_dataset_keys = {
                "pretrain": "pretrain_dataset",
                "eval": "eval_dataset",
            }
            for child_key, flat_key in top_level_dataset_keys.items():
                if child_key in dataset_cfg:
                    expanded[flat_key] = cls._resolve_path(dataset_root, dataset_cfg[child_key])

            task_cfg = dataset_cfg.get("task")
            if task_cfg is not None:
                if not isinstance(task_cfg, dict):
                    raise ValueError("Config key dataset.task must be a mapping")
                task_root = cls._resolve_path(dataset_root, task_cfg.get("root", ""))
                task_dataset_keys = {
                    "identity_conversations": "identity_conversations_dataset",
                    "simple_spelling": "simple_spelling_dataset",
                    "allenai_arc": "allenai_arc_dataset",
                    "openai_gsm8k": "openai_gsm8k_dataset",
                    "openai_humaneval": "openai_humaneval_dataset",
                    "cais_mmlu": "cais_mmlu_dataset",
                    "huggingface_tb_smol_smoltalk": "huggingface_tb_smol_smoltalk_dataset",
                }
                for child_key, flat_key in task_dataset_keys.items():
                    if child_key in task_cfg:
                        expanded[flat_key] = cls._resolve_path(task_root, task_cfg[child_key])

        checkpoint_cfg = data.get("checkpoint")
        if checkpoint_cfg is not None:
            if not isinstance(checkpoint_cfg, dict):
                raise ValueError("Config key checkpoint must be a mapping")
            checkpoint_root = checkpoint_cfg.get("root", "")
            checkpoint_keys = {
                "base": "base_checkpoints_dir",
                "chatsft": "chatsft_checkpoints_dir",
                "chatrl": "chatrl_checkpoints_dir",
            }
            for child_key, flat_key in checkpoint_keys.items():
                if child_key in checkpoint_cfg:
                    expanded[flat_key] = cls._resolve_path(checkpoint_root, checkpoint_cfg[child_key])

        output_cfg = data.get("output")
        if output_cfg is not None:
            if not isinstance(output_cfg, dict):
                raise ValueError("Config key output must be a mapping")
            output_root = output_cfg.get("root", "")
            if output_root:
                expanded["output_dir"] = output_root

            output_keys = {
                "base_eval": "base_eval_dir",
                "chatsft_eval": "chatsft_eval_dir",
                "chatrl_eval": "chatrl_eval_dir",
                "tokenizer": "tokenizer_dir",
                "report": "report_dir",
            }
            for child_key, flat_key in output_keys.items():
                if child_key in output_cfg:
                    expanded[flat_key] = cls._resolve_path(output_root, output_cfg[child_key])

        return expanded

    @classmethod
    def load_from_yaml(cls, config_path: str) -> "GlobalConfig":
        """Load config from a YAML file. Returns a new frozen GlobalConfig instance."""
        try:
            with open(config_path, "r") as file:
                data = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise yaml.YAMLError(f"Invalid YAML in {config_path}") from exc
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError(f"Top-level YAML in {config_path} must be a mapping")
        data = cls._expand_hierarchical_paths(data)
        defaults = cls()
        kwargs = {}
        for f in fields(cls):
            key = f.name
            if key in data:
                value = data[key]
                if type(value) != type(getattr(defaults, key)):
                    raise ValueError(
                        f"Config value {value} for key {key} is of type {type(value)} "
                        f"but should be of type {type(getattr(defaults, key))}"
                    )
                kwargs[key] = value
            else:
                kwargs[key] = getattr(defaults, key)
        valid_keys = {f.name for f in fields(cls)}
        for key in data:
            if key not in valid_keys:
                raise ValueError(f"Config key {key} not found in config class")
        return cls(**kwargs)

    def nice_print(self):
        print("GlobalConfig:")
        print(json.dumps({f.name: getattr(self, f.name) for f in fields(self)}, indent=2))
    


if __name__ == "__main__":
    print(f"{'='*50} before loading {'='*50}")
    GlobalConfig().nice_print()
    global_config = GlobalConfig.load_from_yaml("configs/global.yaml")
    print(f"{'='*50} after loading {'='*50}")
    global_config.nice_print()
    
    # frozen=True, so this will raise an error like this
    # "dataclasses.FrozenInstanceError: cannot assign to field 'device'""
    #config.device = "npu"
