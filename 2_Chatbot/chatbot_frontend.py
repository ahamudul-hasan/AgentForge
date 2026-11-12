import sys
import time
import uuid
from html import escape
from pathlib import Path
import streamlit as st

# Ensure 2_Chatbot directory is in Python path regardless of execution directory
sys.path.append(str(Path(__file__).parent))

from chatbot import stream_chat

# Page configuration
st.set_page_config(page_title="LangGraph Chatbot", page_icon="🤖", layout="centered")

# Custom CSS for a sleek, designable "Thinking" status UI (Light & Dark mode friendly)
st.markdown(
    """
    <style>
    :root {
        --thinking-accent: #6366f1;
        --thinking-bg: linear-gradient(135deg, rgba(99, 102, 241, 0.07), rgba(168, 85, 247, 0.04));
        --thinking-border: rgba(128, 134, 155, 0.25);
        --thinking-step-bg: rgba(128, 134, 155, 0.08);
        --thinking-badge-bg: rgba(99, 102, 241, 0.14);
        --thinking-badge-color: #818cf8;
    }

    /* Thinking status container inside assistant messages */
    [data-testid="stChatMessage"] [data-testid="stExpander"] {
        border: 1px solid var(--thinking-border);
        border-left: 3px solid var(--thinking-accent);
        border-radius: 12px;
        background: var(--thinking-bg);
        margin-bottom: 0.75rem;
        overflow: hidden;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    [data-testid="stChatMessage"] [data-testid="stExpander"]:hover {
        border-color: rgba(99, 102, 241, 0.5);
        box-shadow: 0 2px 10px rgba(99, 102, 241, 0.08);
    }

    [data-testid="stChatMessage"] [data-testid="stExpander"] summary {
        font-size: 0.88rem;
        font-weight: 500;
        padding: 0.45rem 0.85rem;
    }

    /* Individual step cards inside the thinking box */
    .thinking-step {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        padding: 7px 10px;
        margin: 4px 0;
        border-radius: 8px;
        background: var(--thinking-step-bg);
        font-size: 0.83rem;
        line-height: 1.4;
    }

    .thinking-badge {
        display: inline-block;
        padding: 1px 8px;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 600;
        background: var(--thinking-badge-bg);
        color: var(--thinking-badge-color);
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_thinking_step(icon: str, text: str, *badges: str) -> str:
    """Build a styled HTML step row for the thinking status container."""
    badge_html = "".join(
        f'<span class="thinking-badge">{escape(b)}</span>' for b in badges if b
    )
    return (
        f'<div class="thinking-step">'
        f'<span>{icon}</span><span>{text}</span>{badge_html}'
        f'</div>'
    )


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
        st.session_state['thread_id'] = f"thread-{uuid.uuid4().hex[:6]}"
        st.rerun()

# Welcome banner when no messages exist yet
if not st.session_state['message_history']:
    st.info("👋 Hello! I am your AI assistant. Type a message below to start chatting!")

# Render conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        if message['role'] == 'assistant' and message.get('thinking'):
            thinking = message['thinking']
            with st.status(thinking['label'], state="complete", expanded=False):
                for step_html in thinking['steps']:
                    st.markdown(step_html, unsafe_allow_html=True)
        st.markdown(message['content'])

# Chat input at bottom
user_input = st.chat_input('Type your message here...')

if user_input:
    # 1. Add user message to history & display immediately
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    # 2. Stream assistant response with styled thinking status container
    with st.chat_message('assistant'):
        start_time = time.perf_counter()
        thinking_steps: list[str] = []
        status = st.status("🧠 Thinking...", expanded=True)

        init_step = format_thinking_step("🔍", "Routing prompt through LangGraph...")
        thinking_steps.append(init_step)
        status.markdown(init_step, unsafe_allow_html=True)

        def handle_step(metadata: dict):
            node = metadata.get("langgraph_node", "unknown")
            model = metadata.get("ls_model_name", "LLM")
            step_html = format_thinking_step("⚡", "Executing graph node", node, model)
            thinking_steps.append(step_html)
            status.markdown(step_html, unsafe_allow_html=True)
            status.update(label="⚡ Generating response...", state="running")

        try:
            ai_message = st.write_stream(
                stream_chat(user_input, st.session_state['thread_id'], on_step=handle_step)
            )
            elapsed = time.perf_counter() - start_time
            done_step = format_thinking_step("✅", f"Completed in <b>{elapsed:.2f}s</b>")
            thinking_steps.append(done_step)
            status.markdown(done_step, unsafe_allow_html=True)

            final_label = f"✨ Thought for {elapsed:.2f}s"
            status.update(label=final_label, state="complete", expanded=False)

            # 3. Add full AI message + thinking summary to session history
            st.session_state['message_history'].append({
                'role': 'assistant',
                'content': ai_message,
                'thinking': {'label': final_label, 'steps': thinking_steps},
            })
        except Exception as exc:
            status.update(label="❌ Error generating response", state="error", expanded=True)
            st.error(f"Failed to generate response: {exc}")
