"""
Configuration for the nanochat-ascend model.
"""

from dataclasses import dataclass
import yaml
import json

@dataclass
class Config:
    device: str = "cpu"
    enforce_eager: bool = True
    pretrain_dataset: str = ""
    sft_dataset: str = ""
    eval_dataset: str = ""
    output_dir: str = ""

    def load_from_yaml(self, config_path: str):
        with open(config_path, 'r') as file:
            try:
                data = yaml.safe_load(file)
                # yaml config should be exactly the same as the config class
                for key, value in data.items():
                    if key not in self.__dict__:
                        raise ValueError(f"Config key {key} not found in config class")
                    if type(value) != type(getattr(self, key)):
                        raise ValueError(f"Config value {value} for key {key} is of type {type(value)} but should be of type {type(getattr(self, key))}")
                    setattr(self, key, value)
            except yaml.YAMLError as exc:
                print(exc)
    
    def nice_print(self):
        print("Config:")
        print(json.dumps(self.__dict__, indent=2))
    


if __name__ == "__main__":
    config = Config()
    print(f"{'='*50} before loading {'='*50}")
    config.nice_print()
    config.load_from_yaml("configs/local.yaml")
    print(f"{'='*50} after loading {'='*50}")
    config.nice_print()
