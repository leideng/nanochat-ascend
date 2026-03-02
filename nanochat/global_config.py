"""
Configuration for the nanochat-ascend model.
"""

from dataclasses import dataclass, fields
import yaml
import json

@dataclass(frozen=True)
class GlobalConfig:
    device: str = "cpu"
    enforce_eager: bool = True
    pretrain_dataset: str = ""
    sft_dataset: str = ""
    eval_dataset: str = ""
    output_dir: str = ""
    base_checkpoints_dir: str = ""
    chatsft_checkpoints_dir: str = ""
    chatrl_checkpoints_dir: str = ""
    
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
    global_config = GlobalConfig.load_from_yaml("configs/local.yaml")
    print(f"{'='*50} after loading {'='*50}")
    global_config.nice_print()
    
    # frozen=True, so this will raise an error like this
    # "dataclasses.FrozenInstanceError: cannot assign to field 'device'""
    #config.device = "npu"
