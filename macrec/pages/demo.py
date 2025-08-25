import os
import streamlit as st
from loguru import logger

from macrec.pages.task import task_config
from macrec.systems import *
from macrec.utils import task2name, init_api, system2dir, read_json

all_tasks = ['rp', 'sr', 'rr', 'gen', 'chat']

# Available model options
AVAILABLE_MODELS = [
    'gemini-2.0-flash-001',
    'gemini-2.0-flash-lite-001', 
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
    
    # Add CSS to improve sidebar visibility
    st.markdown("""
    <style>
    /* Improve sidebar visibility when collapsed */
    .css-1d391kg {
        width: auto !important;
        min-width: 250px !important;
    }
    
    /* Better checkbox styling */
    .stCheckbox > label {
        font-size: 14px !important;
        font-weight: 500 !important;
    }
    
    /* Improve sidebar spacing */
    .css-1lcbmhc {
        padding-top: 1rem !important;
    }
    
    /* Better button and widget spacing */
    .stSelectbox > label, .stTextInput > label {
        font-size: 13px !important;
        font-weight: 500 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.sidebar.title('🧠 MACRec Demo')
    
    # Model selection
    st.sidebar.markdown("### 🤖 Model Configuration")
    
    use_custom = st.sidebar.checkbox('🎯 Use custom model name', help='Enable to type a custom model name instead of selecting from the list')
    
    if use_custom:
        # Custom model input (full width when enabled)
        custom_model = st.sidebar.text_input(
            'Model name',
            placeholder='e.g., openai/gpt-4o-mini',
            help='Enter any model name supported by your API providers'
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
    
    # Display current model selection with better formatting for sidebar
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
        
        # More compact display for narrow sidebar
        st.sidebar.success(f"{provider_emoji} **{provider_name}**")
        st.sidebar.caption(f"Model: `{model_override}`")
    else:
        st.sidebar.info("💡 Using default models")
    
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
