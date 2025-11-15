import os
import streamlit as st
from loguru import logger
from typing import Optional

from macrec.systems import *
from macrec.utils import task2name, read_json
from macrec.pages.chat import chat_page
from macrec.pages.generation import gen_page

def get_available_datasets():
    """Get all available datasets from the data/ folder."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
    if not os.path.exists(data_dir):
        return ['ml-100k']  # fallback
    
    datasets = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            datasets.append(item)
    
    # Sort datasets with ml-100k first, then alphabetically
    if 'ml-100k' in datasets:
        datasets.remove('ml-100k')
        datasets.sort()
        datasets.insert(0, 'ml-100k')
    else:
        datasets.sort()
    
    return datasets if datasets else ['ml-100k']

def scan_list(config: list) -> bool:
    for i, item in enumerate(config):
        if isinstance(item, dict):
            if not scan_dict(item):
                return False
        elif isinstance(item, list):
            if not scan_list(item):
                return False
        elif isinstance(item, str):
            if os.path.isfile(item) and item.endswith('.json'):
                if not check_json(item):
                    return False
    return True

def scan_dict(config: dict) -> bool:
    for key in config:
        if isinstance(config[key], str):
            if os.path.isfile(config[key]) and config[key].endswith('.json'):
                if not check_json(config[key]):
                    return False
        elif isinstance(config[key], dict):
            if not scan_dict(config[key]):
                return False
        elif isinstance(config[key], list):
            if not scan_list(config[key]):
                return False
    return True

def check_json(config_path: str) -> bool:
    config = read_json(config_path)
    if 'model_path' in config:
        st.markdown(f'`{config_path}` requires `{config["model_path"]}` models.')
        return False
    return scan_dict(config)

def check_config(config_path: str) -> bool:
    return check_json(config_path)

def get_system(system_type: type[System], config_path: str, task: str, dataset: str, model_override: str = None, provider: str = None) -> System:
    logger.debug(f'get_system called with model_override: {model_override}, provider: {provider}')
    system_kwargs = {
        'config_path': config_path,
        'task': task,
        'leak': False,
        'web_demo': True,
        'dataset': dataset
    }

    # Only add model override if both are provided
    if model_override and provider:
        system_kwargs['model_override'] = model_override
        system_kwargs['provider'] = provider
        logger.debug(f'Added model_override to system_kwargs: {model_override} ({provider})')
    else:
        logger.debug('No model override - using agent config files')

    logger.debug(f'Creating system with kwargs: {system_kwargs}')
    return system_type(**system_kwargs)

def task_config(
    task: str,
    system_type: type[System],
    config_path: str,
    model_override: str = None,
    provider: str = None,
    display_model: Optional[str] = None
) -> None:
    logger.debug(f'task_config called with model_override: {model_override} ({provider})')
    st.markdown(f'## `{system_type.__name__}` for {task2name(task)}')

    # Display model info
    if model_override and provider:
        model_label = display_model or model_override
        provider_labels = {
            'openrouter': '**OpenRouter API**',
            'openai': '**OpenAI API**',
            'ollama': '**Ollama (local)**',
            'gemini': '**Gemini API**',
        }
        provider_display = provider_labels.get(provider, f'**{provider.upper()}**')
        st.info(f"Using model: **{model_label}** via {provider_display}")
    else:
        st.info("Using model configuration from agent config files")

    if not check_config(config_path):
        st.error('This config requires resources that are not currently available. Please choose a different configuration.')
        return

    available_datasets = get_available_datasets()
    dataset = st.selectbox('Choose a dataset', available_datasets) if task != 'chat' else 'chat'


    renew = False
    if 'system_type' not in st.session_state or st.session_state.system_type != system_type.__name__:
        renew = True
    elif 'task' not in st.session_state or st.session_state.task != task:
        renew = True
    elif 'config_path' not in st.session_state or st.session_state.config_path != config_path:
        renew = True
    elif 'dataset' not in st.session_state or st.session_state.dataset != dataset:
        renew = True
    elif 'model_override' not in st.session_state or st.session_state.model_override != model_override:
        renew = True
    elif 'provider' not in st.session_state or st.session_state.provider != provider:
        renew = True
    elif 'system' not in st.session_state:
        renew = True
    elif dataset != getattr(st.session_state.system.manager, 'dataset', None):
        renew = True

    if renew:
        system = get_system(system_type, config_path, task, dataset, model_override, provider)
        st.session_state.system_type = system_type.__name__
        st.session_state.task = task
        st.session_state.config_path = config_path
        st.session_state.dataset = dataset
        st.session_state.model_override = model_override
        st.session_state.provider = provider
        st.session_state.system = system
        st.session_state.chat_history = []
        if 'data_sample' in st.session_state:
            del st.session_state.data_sample
    else:
        system = st.session_state.system

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    assert isinstance(st.session_state.chat_history, list)

    if task == 'chat':
        chat_page(system)
    elif task in ['rp', 'sr', 'gen']:
        gen_page(system, task, dataset)
    else:
        raise NotImplementedError

