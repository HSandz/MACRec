import streamlit as st
from loguru import logger

from macrec.systems import System
from macrec.utils import add_chat_message

def chat_page(system: System) -> None:
    for chat in st.session_state.chat_history:
        if isinstance(chat['message'], str):
            st.chat_message(chat['role']).markdown(chat['message'])
        elif isinstance(chat['message'], list):
            with st.chat_message(chat['role']):
                for message in chat['message']:
                    st.markdown(f'{message}')
        else:
            raise ValueError
    logger.debug('Initialization complete!')
    if prompt := st.chat_input():
        add_chat_message('user', prompt)
        with st.chat_message('assistant'):
            st.markdown('#### System is running...')
            try:
                response = system(prompt)
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'message': ['#### System is running...'] + system.web_log
                })
                add_chat_message('assistant', response)
            except Exception as e:
                logger.error(f'Chat system error: {e}')
                error_msg = f'Error: {str(e)}'
                add_chat_message('assistant', error_msg)
        st.rerun()
