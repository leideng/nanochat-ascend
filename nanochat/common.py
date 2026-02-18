"""
Common utilities for nanochat.
"""

import os
import re
import logging
import torch
import torch.distributed as dist

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    def format(self, record):
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        # Format the message
        message = super().format(record)
        # Add color to specific parts of the message
        if levelname == 'INFO':
            # Highlight numbers and percentages
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        return message

def setup_default_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )

setup_default_logging()
logger = logging.getLogger(__name__)

def get_base_dir():
    # co-locate nanochat intermediates with other cached data in ~/.cache (by default)
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir

# The base/pretraining dataset is a set of parquet files.
# You should first download the dataset and saving into a folder
# and then assign the folder to the environment variable NANOCHAT_BASE_DATA_DIR.
def get_base_data_dir():
    if os.environ.get("NANOCHAT_BASE_DATA_DIR"):
        base_data_dir = os.environ.get("NANOCHAT_BASE_DATA_DIR")
    else:
        base_dir = get_base_dir()
        base_data_dir = os.path.join(base_dir, "base_data")
    os.makedirs(base_data_dir, exist_ok=True)
    return base_data_dir

def print0(s="",**kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

def print_banner():
    # Cool DOS Rebel font ASCII banner made with https://manytools.org/hacker-tools/ascii-banner/
    banner = """
                                                                                                        
 #    #   ##   #    #  ####   ####  #    #   ##   #####         ##    ####   ####  ###### #    # #####  
 ##   #  #  #  ##   # #    # #    # #    #  #  #    #          #  #  #      #    # #      ##   # #    # 
 # #  # #    # # #  # #    # #      ###### #    #   #   ##### #    #  ####  #      #####  # #  # #    # 
 #  # # ###### #  # # #    # #      #    # ######   #         ######      # #      #      #  # # #    # 
 #   ## #    # #   ## #    # #    # #    # #    #   #         #    # #    # #    # #      #   ## #    # 
 #    # #    # #    #  ####   ####  #    # #    #   #         #    #  ####   ####  ###### #    # #####                                                                                                         
    """
    print0(banner)

def is_ddp_requested() -> bool:
    """
    True if launched by torchrun (env present), even before init.
    Used to decide whether we *should* initialize a PG.
    """
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))

def is_ddp_initialized() -> bool:
    """
    True if torch.distributed is available and the process group is initialized.
    Used at cleanup to avoid destroying a non-existent PG.
    """
    return dist.is_available() and dist.is_initialized()

def get_dist_info():
    if is_ddp_requested():
        # We rely on torchrun's env to decide if we SHOULD init.
        # (Initialization itself happens in compute init.)
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1

def autodetect_device_type():
    # prefer to use ascend npu if available, otherwise use CPU
    if hasattr(torch, "npu") and torch.npu.is_available():
        device_type = "npu"
    else:
        device_type = "cpu"
    print0(f"Autodetected device type: {device_type}")
    return device_type

def compute_init(device_type="npu"): # npu|cpu
    """Basic initialization that we keep doing over and over, so make common."""

    assert device_type in ["npu", "cpu"], "Invalid device type atm"
    if device_type == "npu":
        assert torch.npu.is_available(), "Your PyTorch installation is not configured for NPU but device_type is 'npu'"
    
    # Reproducibility
    # Note that we set the global seeds here, but most of the code uses explicit rng objects.
    # The only place where global rng might be used is nn.Module initialization of the model weights.
    torch.manual_seed(42)
    if device_type == "npu":
        torch.npu.manual_seed(42)
    # skipping full reproducibility for now, possibly investigate slowdown later
    # torch.use_deterministic_algorithms(True)

    # Precision
    if device_type == "npu":
        torch.backends.fp32_precision = "tf32" # uses tf32 instead of fp32 for matmuls

    # Distributed setup: Distributed Data Parallel (DDP), optional, and requires NPU
    is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if is_ddp_requested and device_type == "npu":
        device = torch.device("npu", ddp_local_rank)
        torch.npu.set_device(device)  # make npu default to this device
        dist.init_process_group(backend="hccl", rank=ddp_rank, world_size=ddp_world_size, device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type) # cpu

    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")

    return is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size, device

def compute_cleanup():
    """Companion function to compute_init, to clean things up before script exit"""
    if is_ddp_initialized():
        dist.destroy_process_group()

class DummyWandb:
    """Useful if we wish to not use wandb but have all the same signatures"""
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass

# hardcoded BF16 peak flops for Ascend NPUs
def get_peak_flops(device_name: str) -> float:
    name = device_name.lower()

    _PEAK_FLOPS_TABLE = (
        # Ascend NPU
        (["910c"], 256e12),
        (["910b"], 256e12),
    )
    for patterns, flops in _PEAK_FLOPS_TABLE:
        if all(p in name for p in patterns):
            return flops

    # Unknown device - return inf so MFU shows as 0% rather than a wrong guess
    logger.warning(f"Peak flops undefined for: {device_name}, MFU will show as 0%")
    return float('inf')
