# Description: Initialization functions.

import os
import random
from typing import Dict, Iterator, Tuple

import numpy as np
from loguru import logger


def _normalize_provider_name(value: str | None) -> str:
    """Return a lowercase provider alias with whitespace removed."""
    return (value or '').strip().lower()


def _provider_aliases(name: str, config: dict) -> set[str]:
    """Return all aliases that should match this provider entry."""
    aliases = {
        _normalize_provider_name(name),
        _normalize_provider_name(config.get('type')),
        _normalize_provider_name(config.get('provider')),
    }
    for alias in config.get('aliases', []):
        aliases.add(_normalize_provider_name(alias))
    # Remove empty strings
    return {alias for alias in aliases if alias}


def iter_provider_settings(api_config: dict) -> Iterator[Tuple[str, Dict]]:
    """Yield `(provider, provider_config)` pairs from api_config.

    Supports both legacy single-provider configs and the new multi-provider layout.
    """
    if not isinstance(api_config, dict):
        raise ValueError("API configuration must be a dictionary")

    if 'providers' in api_config:
        providers: dict = api_config.get('providers', {})
        if not providers:
            raise ValueError("Multi-provider API config requires at least one entry in 'providers'")

        default_entry = api_config.get('default_provider')
        yielded: set[str] = set()

        def _yield_entry(entry_name: str) -> Iterator[Tuple[str, Dict]]:
            cfg = providers[entry_name].copy()
            provider = _normalize_provider_name(cfg.get('type') or entry_name)
            cfg.setdefault('provider', provider)
            yielded.add(entry_name)
            yield provider, cfg

        if default_entry and default_entry in providers:
            yield from _yield_entry(default_entry)

        for name in providers:
            if name not in yielded:
                yield from _yield_entry(name)
        return

    # Legacy single-provider / mixed configuration
    provider = _normalize_provider_name(api_config.get('provider', 'openrouter'))
    cfg = api_config.copy()
    cfg.setdefault('provider', provider or 'openrouter')

    yield provider or 'openrouter', cfg


def get_provider_settings(api_config: dict, provider_name: str | None = None) -> Tuple[str, Dict]:
    """Return `(provider, provider_config)` for the requested provider."""
    target = _normalize_provider_name(provider_name)

    if 'providers' in api_config:
        providers: dict = api_config.get('providers', {})
        if not providers:
            raise ValueError("Multi-provider API config requires at least one entry in 'providers'")

        # Try explicit target first
        if target:
            for name, cfg in providers.items():
                if target in _provider_aliases(name, cfg):
                    result = cfg.copy()
                    provider = _normalize_provider_name(result.get('type') or name)
                    result.setdefault('provider', provider)
                    return provider, result
            raise ValueError(f"Provider '{provider_name}' not found in API configuration.")

        # Fall back to default
        default_entry = api_config.get('default_provider')
        if default_entry and default_entry in providers:
            result = providers[default_entry].copy()
            provider = _normalize_provider_name(result.get('type') or default_entry)
            result.setdefault('provider', provider)
            return provider, result

        # Otherwise take first entry
        name, cfg = next(iter(providers.items()))
        result = cfg.copy()
        provider = _normalize_provider_name(result.get('type') or name)
        result.setdefault('provider', provider)
        return provider, result

    # Legacy configuration
    provider = _normalize_provider_name(api_config.get('provider', 'openrouter'))
    cfg = api_config.copy()

    # No mixed provider support

    if target and provider != target:
        raise ValueError(f"Requested provider '{target}' does not match config provider '{provider}'.")

    cfg.setdefault('provider', provider or 'openrouter')
    return cfg['provider'], cfg


def init_gemini_api(api_key: str):
    """Initialize Gemini API.

    Args:
        `api_key` (`str`): Gemini API key.
    """
    if not api_key:
        raise ValueError("Gemini API key cannot be empty")

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
    except ImportError as exc:
        raise ImportError(
            "google-generativeai package is required for Gemini API. Install it with: pip install google-generativeai"
        ) from exc


def init_openrouter_api(api_key: str):
    """Initialize OpenRouter API.

    Args:
        `api_key` (`str`): OpenRouter API key.
    """
    if not api_key:
        raise ValueError("OpenRouter API key cannot be empty")


def init_openai_api(api_key: str, base_url: str | None = None):
    """Initialize OpenAI API.

    Args:
        `api_key` (`str`): OpenAI API key.
        `base_url` (`str|None`): Optional API base URL (proxy). When provided,
            the client will be instantiated with this base URL to avoid
            contacting OpenAI's default endpoint directly.
    """
    if not api_key:
        raise ValueError("OpenAI API key cannot be empty")

    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "openai package is required for OpenAI API. Install it with: pip install openai>=1.0.0"
        ) from exc

    os.environ['OPENAI_API_KEY'] = api_key
    # Instantiate client once to validate configuration (no network call).
    # Pass base_url when available to ensure proxy-only usage.
    if base_url:
        OpenAI(api_key=api_key, base_url=base_url)
    else:
        OpenAI(api_key=api_key)


def init_api(api_config: dict):
    """Initialize API providers defined in the configuration."""
    initialized = False

    for provider, provider_cfg in iter_provider_settings(api_config):
        if provider == 'openrouter':
            api_key = provider_cfg.get('api_key') or provider_cfg.get('openrouter_api_key')
            if api_key:
                init_openrouter_api(api_key)
                initialized = True
            else:
                logger.info("Skip OpenRouter provider: api_key is empty.")
        elif provider == 'openai':
            api_key = provider_cfg.get('api_key') or provider_cfg.get('openai_api_key') or provider_cfg.get('openai_key')
            base_url = provider_cfg.get('base_url')
            if api_key:
                init_openai_api(api_key, base_url=base_url)
                initialized = True
            else:
                logger.info("Skip OpenAI provider: api_key is empty.")
        elif provider == 'gemini':
            api_key = provider_cfg.get('api_key') or provider_cfg.get('gemini_api_key')
            if api_key:
                init_gemini_api(api_key)
                initialized = True
            else:
                logger.info("Skip Gemini provider: api_key is empty.")
        elif provider == 'ollama':
            # Local models do not require initialization
            base_url = provider_cfg.get('base_url')
            if base_url:
                os.environ.setdefault('OLLAMA_BASE_URL', base_url)
            logger.debug(f"Ollama provider configured at {base_url or 'http://localhost:11434'}")
        else:
            logger.warning(f"Unsupported API provider '{provider}', skipping.")

    if not initialized:
        logger.warning("No cloud API providers were initialized. Only local models may be available.")


def init_all_seeds(seed: int = 0) -> None:
    """Initialize all seeds.

    Args:
        `seed` (`int`, optional): Random seed. Defaults to `0`.
    """
    random.seed(seed)
    np.random.seed(seed)
    # Note: torch seeding removed since torch is no longer a dependency
