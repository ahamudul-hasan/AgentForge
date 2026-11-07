import sys
from pathlib import Path
import streamlit as st

# Ensure 2_Chatbot directory is in Python path regardless of execution directory
sys.path.append(str(Path(__file__).parent))

from chatbot import chatbot
from langchain_core.messages import HumanMessage

# Page configuration
st.set_page_config(page_title="LangGraph Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 LangGraph AI Chatbot")
st.caption("Powered by LangGraph, Google Gemini & Streamlit")

# Session state initialization
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = 'thread-1'

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Chat Settings")
    st.write(f"**Current Thread:** `{st.session_state['thread_id']}`")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state['message_history'] = []
        import uuid
        st.session_state['thread_id'] = f"thread-{uuid.uuid4().hex[:6]}"
        st.rerun()

CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

# Welcome banner when no messages exist yet
if not st.session_state['message_history']:
    st.info("👋 Hello! I am your AI assistant. Type a message below to start chatting!")

# Render conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Chat input at bottom
user_input = st.chat_input('Type your message here...')

if user_input:
    # 1. Add user message to history & display immediately
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    # 2. Invoke LangGraph with visual loading spinner
    with st.chat_message('assistant'):
        with st.spinner("Thinking..."):
            response = chatbot.invoke(
                {'messages': [HumanMessage(content=user_input)]}, 
                config=CONFIG
            )
            last_msg = response['messages'][-1]
            ai_message = getattr(last_msg, 'text', None) or (
                last_msg.content[0]['text'] if isinstance(last_msg.content, list) else str(last_msg.content)
            )
            st.markdown(ai_message)

    # 3. Add AI message to session history
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
