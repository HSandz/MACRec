import os
import streamlit as st
from loguru import logger

from macrec.pages.task import task_config
from macrec.systems import SYSTEMS
from macrec.utils import task2name, init_api, system2dir, read_json

all_tasks = ['rp', 'sr', 'gen', 'chat']

# Available model options - Updated to reflect current supported models
AVAILABLE_MODELS = [
    'google/gemini-2.0-flash-001',
    'google/gemini-2.0-flash-lite-001',
    'google/gemini-1.5-flash', 
    'google/gemini-1.5-pro',
    'openai/gpt-4o',
    'openai/gpt-4o-mini',
    'openai/gpt-3.5-turbo',
    'anthropic/claude-3-5-sonnet',
    'anthropic/claude-3-5-haiku',
    'meta-llama/llama-3.1-8b-instruct',
    'meta-llama/llama-3.1-70b-instruct',
    'meta-llama/llama-3.2-1b-instruct',
    'meta-llama/llama-3.2-3b-instruct',
    'mistralai/mistral-7b-instruct',
    'deepseek/deepseek-r1',
    'qwen/qwen-2.5-72b-instruct'
]

def demo():
    init_api(read_json('config/api-config.json'))
    st.set_page_config(
        page_title="MACRec Demo",
        page_icon="🧠",
        layout="wide",
    )

    st.sidebar.title('MACRec Demo')
    # Model selection
    st.sidebar.markdown("Model Configuration")
    
    use_custom = st.sidebar.checkbox('Use custom model name', help='Enable to type a custom model name instead of selecting from the list')
    
    if use_custom:
        # Custom model input (full width when enabled)
        custom_model = st.sidebar.text_input(
            'Model name',
            placeholder='e.g., openai/gpt-4o-mini, google/gemini-2.0-flash-001',
            help='Enter any model name supported by OpenRouter or direct API providers'
        )
        model_override = custom_model if custom_model.strip() else 'google/gemini-2.0-flash-001'
        logger.debug(f'Using custom model: {model_override}')
    else:
        # Selectbox for predefined models (full width when enabled)
        selected_model = st.sidebar.selectbox(
            'Choose a model',
            options=[''] + AVAILABLE_MODELS,
            format_func=lambda x: 'Select a model...' if x == '' else x,
            help='Select from available models'
        )
        model_override = selected_model if selected_model != '' else 'google/gemini-2.0-flash-001'
        logger.debug(f'Using selected model: {model_override}')
    
    logger.debug(f'Final model_override: {model_override}')

    
    st.sidebar.markdown("---")
    
    # choose a system
    system_type = st.sidebar.radio('Choose a system', SYSTEMS, format_func=lambda x: x.__name__)
    # choose the config
    config_dir = os.path.join('config', 'systems', system2dir(system_type.__name__))
    config_files = os.listdir(config_dir)
    config_file = st.sidebar.selectbox('Choose a config file', config_files)
    config = read_json(os.path.join(config_dir, config_file))
    assert 'supported_tasks' in config, f'The config file {config_file} should contain the field "supported_tasks".'
    supported_tasks = config['supported_tasks']
    supported_tasks = [task for task in supported_tasks if task in system_type.supported_tasks()]
    # choose a task
    task = st.sidebar.radio('Choose a task', all_tasks, format_func=task2name)
    
    # Show information about deprecated systems
    st.sidebar.markdown("---")
    st.sidebar.markdown("**💡 Note:** The following systems have been deprecated:")
    st.sidebar.markdown("- `react` → Use `collaboration` system")
    st.sidebar.markdown("- `reflection` → Use `collaboration` with reflection")
    st.sidebar.markdown("- `analyse` → Use `collaboration` with analysis")
    
    if task not in supported_tasks:
        st.error(f'The task {task2name(task)} is not supported by the system `{system_type.__name__}` with the config file `{config_file}`. Supported tasks: {", ".join([task2name(t) for t in supported_tasks])}')
        return
    task_config(task=task, system_type=system_type, config_path=os.path.join(config_dir, config_file), model_override=model_override)
