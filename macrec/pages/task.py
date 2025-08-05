import os
import streamlit as st
from loguru import logger

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
    if 'model_type' in config and config['model_type'] == 'opensource':
        assert 'model_path' in config, 'model_path is required for OpenSource models'
        st.markdown(f'`{config_path}` requires `{config["model_path"]}` models.')
        return False
    if 'model_path' in config:
        st.markdown(f'`{config_path}` requires `{config["model_path"]}` models.')
        return False
    return scan_dict(config)

def check_config(config_path: str) -> bool:
    import torch
    if torch.cuda.is_available():
        return True
    else:
        return check_json(config_path)

def get_system(system_type: type[System], config_path: str, task: str, dataset: str) -> System:
    return system_type(config_path=config_path, task=task, leak=False, web_demo=True, dataset=dataset)

def task_config(task: str, system_type: type[System], config_path: str) -> None:
    st.markdown(f'## `{system_type.__name__}` for {task2name(task)}')
    checking = check_config(config_path)
    if not checking:
        st.error('This config file requires OpenSource models, which are not supported in this machine (without cuda toolkit).')
        return
    
    # Dynamically get available datasets
    available_datasets = get_available_datasets()
    dataset = st.selectbox('Choose a dataset', available_datasets) if task != 'chat' else 'chat'
    renew = False
    if 'system_type' not in st.session_state:
        logger.debug(f'New system type: {system_type.__name__}')
        st.session_state.system_type = system_type.__name__
        renew = True
    elif st.session_state.system_type != system_type.__name__:
        logger.debug(f'Change system type: {system_type.__name__}')
        st.session_state.system_type = system_type.__name__
        renew = True
    elif 'task' not in st.session_state:
        logger.debug(f'New task: {task}')
        st.session_state.task = task
        renew = True
    elif st.session_state.task != task:
        logger.debug(f'Change task: {task}')
        st.session_state.task = task
        renew = True
    elif 'config_path' not in st.session_state:
        logger.debug(f'New config path: {config_path}')
        st.session_state.config_path = config_path
        renew = True
    elif st.session_state.config_path != config_path:
        logger.debug(f'Change config path: {config_path}')
        st.session_state.config_path = config_path
        renew = True
    elif 'dataset' not in st.session_state:
        logger.debug(f'New dataset: {dataset}')
        st.session_state.dataset = dataset
        renew = True
    elif st.session_state.dataset != dataset:
        logger.debug(f'Change dataset: {dataset}')
        st.session_state.dataset = dataset
        renew = True
    elif 'system' not in st.session_state:
        logger.debug('New system')
        renew = True
    elif dataset != st.session_state.system.manager.dataset:
        logger.debug(f'Change dataset: {dataset}')
        st.session_state.dataset = dataset
        renew = True
    if renew:
        system = get_system(system_type, config_path, task, dataset)
        st.session_state.system_type = system_type.__name__
        st.session_state.task = task
        st.session_state.config_path = config_path
        st.session_state.dataset = dataset
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
    elif task == 'rp' or task == 'sr' or task == 'gen':
        gen_page(system, task, dataset)
    else:
        raise NotImplementedError
