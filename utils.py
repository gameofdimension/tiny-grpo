import os
import random

import numpy as np
import torch
import torch.distributed as dist


def set_random_seed(seed: int = 42):
    """
    Set the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use for random number generation.

    Returns:
        None

    Explanation:
        1. Sets seed for Python's built-in random module for basic random operations.
        2. Sets seed for NumPy, ensuring consistent random number generation in array operations.
        3. Sets seed for PyTorch CPU operations.
        4. If CUDA is available, sets seed for all GPU devices.
        5. Configures cuDNN to ensure deterministic behavior:
           - Sets deterministic flag to True, ensuring reproducible results.
           - Disables benchmarking to prevent algorithm selection based on hardware.

    Note:
        Setting deterministic behavior may impact performance but ensures consistent results
        across multiple runs, which is crucial for debugging and research.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_main_process():
    try:
        if dist.get_rank() == 0:
            return True
        else:
            return False
    except Exception:
        return True


def cleanup():
    dist.destroy_process_group()


def init_distributed():

    # Initializes the distributed backend
    # which will take care of sychronizing nodes/GPUs
    dist_url = "env://"  # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl", init_method=dist_url, world_size=world_size, rank=rank,
        device_id=torch.device(f"cuda:{torch.cuda.current_device()}"),
    )

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    return world_size, rank, local_rank


def rank0_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)
