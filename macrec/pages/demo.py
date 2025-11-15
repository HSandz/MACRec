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

def _normalize_model_for_provider(model: str, provider: str) -> str:
    """Strip provider prefixes when using direct APIs."""
    cleaned = (model or '').strip()
    if not cleaned:
        return cleaned
    if provider == 'openai' and '/' in cleaned:
        prefix, suffix = cleaned.split('/', 1)
        if prefix.lower() == 'openai':
            return suffix
    if provider == 'ollama' and cleaned.startswith('ollama/'):
        return cleaned.split('/', 1)[1]
    return cleaned

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

    provider_choices = [
        ('Use agent config', None),  # New option: no override
        ('Hosted via OpenRouter', 'openrouter'),
        ('Direct OpenAI', 'openai'),
        ('Local Ollama', 'ollama'),
    ]
    provider_label = st.sidebar.selectbox(
        'LLM provider',
        options=[label for label, _ in provider_choices],
        index=0,
        help='Choose where the model is served from, or use agent config files'
    )
    provider = dict(provider_choices)[provider_label]

    # Only show model selection if a provider is chosen
    if provider is None:
        st.sidebar.info('ℹ️ Using model configuration from agent config files')
        model_override = None
        model_display = 'Agent Config'
    else:
        default_models = {
            'openrouter': 'google/gemini-2.0-flash-001',
            'openai': 'gpt-4o-mini',
            'ollama': 'llama3.2:1b',
        }
        fallback_model = default_models.get(provider, 'google/gemini-2.0-flash-001')

        use_custom = st.sidebar.checkbox('Use custom model name', help='Enable to type a custom model name instead of selecting from the list')

        if use_custom:
            # Custom model input (full width when enabled)
            custom_model = st.sidebar.text_input(
                'Model name',
                placeholder='e.g., openai/gpt-4o-mini, google/gemini-2.0-flash-001, llama3.2:1b',
                help='Enter any model name supported by the selected provider'
            )
            model_display = custom_model.strip() or fallback_model
            logger.debug(f'Using custom model: {model_display}')
        else:
            # Selectbox for predefined models (full width when enabled)
            selected_model = st.sidebar.selectbox(
                'Choose a model',
                options=[''] + AVAILABLE_MODELS,
                format_func=lambda x: 'Select a model...' if x == '' else x,
                help='Select from available models'
            )
            model_display = selected_model if selected_model else fallback_model
            logger.debug(f'Using selected model: {model_display}')

        model_override = _normalize_model_for_provider(model_display, provider) or fallback_model
        logger.debug(f'Final model_override: {model_override} ({provider})')

    
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
        st.error(f'The task {task2name(task)} is not supported by the system `{system_type.__name__}` with the config file `{config_file}`. Supported tasks: {", ".join([task2name(t) for t in supported_tasks])}')
        return
    task_config(
        task=task,
        system_type=system_type,
        config_path=os.path.join(config_dir, config_file),
        model_override=model_override,
        provider=provider,
        display_model=model_display
    )
