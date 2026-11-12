import sys
import time
import uuid
from html import escape
from pathlib import Path
import streamlit as st

# Ensure 2_Chatbot directory is in Python path regardless of execution directory
sys.path.append(str(Path(__file__).parent))

from chatbot import get_thread_history, stream_chat

# Page configuration
st.set_page_config(page_title="LangGraph Chatbot", page_icon="🤖", layout="centered")

# Custom CSS for sleek ChatGPT/Gemini sidebar & designable "Thinking" status UI
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

    /* ChatGPT / Gemini style left sidebar chat list */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stBaseButton-secondary"],
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stBaseButton-primary"] {
        justify-content: flex-start;
        text-align: left;
        border-radius: 10px;
        padding-left: 0.65rem;
        padding-right: 0.65rem;
    }

    /* Light blue "New Chat" button */
    .st-key-new_chat_btn button {
        background-color: #93c5fd !important;
        border-color: #60a5fa !important;
        color: #0f172a !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        transition: background-color 0.15s ease, border-color 0.15s ease;
    }
    .st-key-new_chat_btn button:hover {
        background-color: #60a5fa !important;
        border-color: #3b82f6 !important;
        color: #0f172a !important;
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


def generate_thread_id() -> str:
    """Create a unique conversation thread identifier."""
    return str(uuid.uuid4())


def new_chat() -> None:
    """Start a fresh chat thread unless the current thread is already empty."""
    current_id = st.session_state['thread_id']
    if not st.session_state['message_history'] and current_id in st.session_state['chat_threads']:
        return
    new_id = generate_thread_id()
    st.session_state['thread_id'] = new_id
    st.session_state['chat_threads'].insert(0, new_id)
    st.session_state['message_history'] = []


def switch_thread(thread_id: str) -> None:
    """Load an existing thread's history from LangGraph checkpointer."""
    st.session_state['thread_id'] = thread_id
    history = get_thread_history(thread_id)
    thinking_map = st.session_state['thinking_cache'].get(thread_id, {})
    for idx, msg in enumerate(history):
        if msg['role'] == 'assistant' and idx in thinking_map:
            msg['thinking'] = thinking_map[idx]
    st.session_state['message_history'] = history


def delete_thread(thread_id: str) -> None:
    """Remove a thread from the sidebar history list."""
    if thread_id in st.session_state['chat_threads']:
        st.session_state['chat_threads'].remove(thread_id)
    st.session_state['thread_titles'].pop(thread_id, None)
    st.session_state['thinking_cache'].pop(thread_id, None)

    if st.session_state['thread_id'] == thread_id:
        if st.session_state['chat_threads']:
            switch_thread(st.session_state['chat_threads'][0])
        else:
            new_id = generate_thread_id()
            st.session_state['thread_id'] = new_id
            st.session_state['chat_threads'] = [new_id]
            st.session_state['message_history'] = []


# Session state initialization
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = [st.session_state['thread_id']]

if 'thread_titles' not in st.session_state:
    st.session_state['thread_titles'] = {}

if 'thinking_cache' not in st.session_state:
    st.session_state['thinking_cache'] = {}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Left Sidebar (ChatGPT / Gemini style navigation)
with st.sidebar:
    st.title("💬 AgentForge")
    if st.button("➕ New Chat", key="new_chat_btn", type="primary", use_container_width=True):
        new_chat()
        st.rerun()

    st.divider()
    st.caption("Recent Chats")

    for tid in list(st.session_state['chat_threads']):
        title = st.session_state['thread_titles'].get(tid, "New Chat")
        is_active = tid == st.session_state['thread_id']
        col_chat, col_del = st.columns([0.82, 0.18], gap="small")
        with col_chat:
            if st.button(
                f"{'💬' if is_active else '💭'} {title}",
                key=f"chat_{tid}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                if not is_active:
                    switch_thread(tid)
                    st.rerun()
        with col_del:
            if st.button("✕", key=f"del_{tid}", help="Delete chat", use_container_width=True):
                delete_thread(tid)
                st.rerun()

# Main Chat Header
st.title("🤖 LangGraph AI Chatbot")
st.caption("Powered by LangGraph, Google Gemini & Streamlit")

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
    active_tid = st.session_state['thread_id']
    is_first_message = len(st.session_state['message_history']) == 0

    # Auto-name the thread from the first user message & move to top of Recent Chats
    if active_tid not in st.session_state['thread_titles']:
        clean_title = " ".join(user_input.split())
        st.session_state['thread_titles'][active_tid] = (
            clean_title[:28] + "..." if len(clean_title) > 28 else clean_title
        )
    if active_tid in st.session_state['chat_threads']:
        st.session_state['chat_threads'].remove(active_tid)
    st.session_state['chat_threads'].insert(0, active_tid)

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
                stream_chat(user_input, active_tid, on_step=handle_step)
            )
            elapsed = time.perf_counter() - start_time
            done_step = format_thinking_step("✅", f"Completed in <b>{elapsed:.2f}s</b>")
            thinking_steps.append(done_step)
            status.markdown(done_step, unsafe_allow_html=True)

            final_label = f"✨ Thought for {elapsed:.2f}s"
            status.update(label=final_label, state="complete", expanded=False)

            thinking_data = {'label': final_label, 'steps': thinking_steps}

            # 3. Save assistant message + thinking summary to session & cache
            msg_index = len(st.session_state['message_history'])
            st.session_state['thinking_cache'].setdefault(active_tid, {})[msg_index] = thinking_data
            st.session_state['message_history'].append({
                'role': 'assistant',
                'content': ai_message,
                'thinking': thinking_data,
            })

            # Refresh UI after first message so sidebar title & welcome banner update immediately
            if is_first_message:
                st.rerun()
        except Exception as exc:
            status.update(label="❌ Error generating response", state="error", expanded=True)
            st.error(f"Failed to generate response: {exc}")
