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

def init_openrouter_api(api_key: str):
    """Initialize OpenRouter API.

    Args:
        `api_key` (`str`): OpenRouter API key.
    """
    # OpenRouter doesn't require global initialization like Gemini
    # The API key is used directly in the OpenRouterLLM class
    if not api_key:
        raise ValueError("OpenRouter API key cannot be empty")

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
    elif provider == 'openrouter':
        if 'api_key' in api_config:
            init_openrouter_api(api_config['api_key'])
        else:
            raise ValueError("OpenRouter API configuration requires 'api_key'")
    elif provider == 'mixed':
        # Support for mixed providers - initialize both if both API keys are present
        if 'gemini_api_key' in api_config:
            init_gemini_api(api_config['gemini_api_key'])
        if 'openrouter_api_key' in api_config:
            init_openrouter_api(api_config['openrouter_api_key'])
        if 'gemini_api_key' not in api_config and 'openrouter_api_key' not in api_config:
            raise ValueError("Mixed API configuration requires at least one of 'gemini_api_key' or 'openrouter_api_key'")
    else:
        raise ValueError(f"Unsupported API provider: {provider}. Supported providers are 'gemini', 'openrouter', and 'mixed'.")

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
