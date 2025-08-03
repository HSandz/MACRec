# Description: Initialization functions.

import os
import random
import numpy as np
import torch

def init_gemini_api(api_key: str):
    """Initialize Gemini API.

    Args:
        `api_key` (`str`): Gemini API key.
    """
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
    except ImportError:
        raise ImportError("google-generativeai package is required for Gemini API. Install it with: pip install google-generativeai")

def init_api(api_config: dict):
    """Initialize API based on configuration.
    
    Args:
        `api_config` (`dict`): API configuration, should contain `provider` and appropriate keys.
    """
    provider = api_config.get('provider', 'gemini').lower()
    
    if provider == 'gemini':
        if 'api_key' in api_config:
            init_gemini_api(api_config['api_key'])
        else:
            raise ValueError("Gemini API configuration requires 'api_key'")
    else:
        raise ValueError(f"Unsupported API provider: {provider}. Only 'gemini' is supported.")

def init_all_seeds(seed: int = 0) -> None:
    """Initialize all seeds.

    Args:
        `seed` (`int`, optional): Random seed. Defaults to `0`.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
