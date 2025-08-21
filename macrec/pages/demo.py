import os
import streamlit as st
from loguru import logger

from macrec.pages.task import task_config
from macrec.systems import *
from macrec.utils import task2name, init_api, system2dir, read_json

all_tasks = ['rp', 'sr', 'rr', 'gen', 'chat']

# Available model options
AVAILABLE_MODELS = [
    'gemini-2.0-flash',
    'gemini-2.0-flash-lite', 
    'openai/gpt-oss-20b:free',
    'z-ai/glm-4.5-air:free',
    'deepseek/deepseek-r1-0528:free',
    'meta-llama/llama-3.1-8b-instruct',
    'meta-llama/llama-3.1-70b-instruct',
    'openai/gpt-4o',
    'anthropic/claude-3-5-sonnet',
    'mistralai/mistral-7b-instruct'
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
    st.sidebar.markdown("### 🤖 Model Configuration")
    
    # Create two columns for model selection UI
    col1, col2 = st.sidebar.columns([3, 1])
    
    with col1:
        # Text input for custom model (with selectbox options)
        selected_model = st.selectbox(
            'Choose a model',
            options=[''] + AVAILABLE_MODELS,
            format_func=lambda x: 'Select or type custom model...' if x == '' else x,
            help='Select from available models or type a custom model name below'
        )
    
    with col2:
        # Option to use custom model
        use_custom = st.checkbox('Custom', help='Enable to type a custom model name')
    
    # Custom model input
    if use_custom:
        custom_model = st.sidebar.text_input(
            'Custom model name',
            placeholder='e.g., openai/gpt-4o-mini',
            help='Enter any model name supported by your API providers'
        )
        model_override = custom_model if custom_model.strip() else None
        logger.debug(f'Using custom model: {model_override}')
    else:
        model_override = selected_model if selected_model != '' else None
        logger.debug(f'Using selected model: {model_override}')
    
    logger.debug(f'Final model_override: {model_override}')
    
    # Display current model selection
    if model_override:
        if model_override.startswith('gemini'):
            provider_emoji = '🟢'
            provider_name = 'Gemini'
        elif '/' in model_override or any(x in model_override.lower() for x in ['gpt', 'claude', 'llama', 'mistral', 'openai', 'anthropic']):
            provider_emoji = '🔴'
            provider_name = 'OpenRouter'
        else:
            provider_emoji = '⚪'
            provider_name = 'Auto-detect'
        
        st.sidebar.info(f"{provider_emoji} Using **{model_override}** via {provider_name}")
    else:
        st.sidebar.info("💡 Using default models from agent configurations")
    
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
    if task not in supported_tasks:
        st.error(f'The task {task2name(task)} is not supported by the system `{system_type.__name__}` with the config file `{config_file}`.')
        return
    task_config(task=task, system_type=system_type, config_path=os.path.join(config_dir, config_file), model_override=model_override)
