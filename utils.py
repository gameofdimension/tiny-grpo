# !pip install tf-keras # for some reason, Hugging Face cannot work without it
# !pip install flash-attn # FlashAttention2
# !pip install wandb # Weights and Biases
# !pip install 'accelerate>=0.26.0'
# !pip install transformers # Hugging Face Transformers API
# !pip install datasets # Hugging Face Datasets API

# Import necessary libraries
# Basic Python libraries for various operations
import random
import copy
import re
import os
import numpy as np
import wandb

# PyTorch and related libraries for deep learning
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# Hugging Face libraries for transformer models
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

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

# Call the function to set random seed for reproducibility
set_random_seed(42)

# Set environment variables for Weights & Biases (wandb) logging
os.environ["WANDB_API_KEY"] = "USE YOUR KEY"
os.environ["WANDB_PROJECT"] = "GRPO-Qwen-1.5-Instruct-Multi-GPU"