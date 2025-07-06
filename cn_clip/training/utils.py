import os
import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True):
    """
    Seed everything for reproducibility.

    Args:
        seed (int): The seed to use.
        deterministic (bool): If True, sets CUDA/cuDNN into deterministic mode
                              (can slow down performance). Default: True.
    """
    # Python built-in RNG
    random.seed(seed)
    # Numpy RNG
    np.random.seed(seed)
    # Torch RNG (CPU)
    torch.manual_seed(seed)
    # Torch RNG (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Hash seed for Python in some cases (e.g. hashing order in dicts)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # Make cuDNN deterministic (possibly slower)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For PyTorch ≥1.8, you can also enforce
        # torch.use_deterministic_algorithms(True)