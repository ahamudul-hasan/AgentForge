from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Iterator, Callable
from langchain_core.messages import BaseMessage, HumanMessage, AIMessageChunk, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import sqlite3
import requests
import os

load_dotenv()

os.environ['LANGSMITH_PROJECT'] = 'chatbot'

# 1. LLM
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", streaming=True)


# 2. Tools
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "C9PE94QUEW9VWGFM")
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    r = requests.get(url)
    return r.json()


tools = [search_tool, get_stock_price, calculator]
llm_with_tools = llm.bind_tools(tools)


# 3. State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# 4. Nodes
def chat_node(state: ChatState):
    """LLM node that may answer directly or call tools."""
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)


# 5. Checkpointer & Graph
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


# 6. Streaming & Thread Helpers
def stream_chat(
    user_input: str,
    thread_id: str,
    on_step: Callable[[dict], None] | None = None,
) -> Iterator[str]:
    """Stream assistant response tokens for a given user input and thread ID."""
    config = {'configurable': {'thread_id': thread_id}}
    current_node = None
    for message_chunk, metadata in chatbot.stream(
        {'messages': [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="messages",
    ):
        node = metadata.get("langgraph_node")
        if node and node != current_node:
            current_node = node
            if on_step:
                meta = dict(metadata)
                if isinstance(message_chunk, ToolMessage):
                    meta["tool_name"] = getattr(message_chunk, "name", None)
                elif hasattr(message_chunk, "tool_calls") and message_chunk.tool_calls:
                    meta["tool_name"] = message_chunk.tool_calls[0].get("name")
                on_step(meta)
        elif on_step and hasattr(message_chunk, "tool_calls") and message_chunk.tool_calls:
            meta = dict(metadata)
            meta["tool_name"] = message_chunk.tool_calls[0].get("name")
            on_step(meta)

        if isinstance(message_chunk, AIMessageChunk) and message_chunk.text:
            yield str(message_chunk.text)


def get_thread_history(thread_id: str) -> list[dict]:
    """Load saved message history for a thread from the LangGraph checkpointer."""
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    messages = state.values.get('messages', []) if state and state.values else []
    history = []
    for msg in messages:
        # Skip tool response messages and non-content messages in visible UI history
        if isinstance(msg, ToolMessage) or getattr(msg, 'type', None) == 'tool':
            continue

        role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
        content = getattr(msg, 'text', None)
        if not content:
            if isinstance(msg.content, list):
                text_parts = [
                    p.get('text', '')
                    for p in msg.content
                    if isinstance(p, dict) and 'text' in p
                ]
                content = "".join(text_parts)
            else:
                content = str(msg.content or "")

        content = content.strip()
        if not content:
            continue

        history.append({'role': role, 'content': content})
    return history


def retrieve_all_threads() -> list[str]:
    """Retrieve all distinct thread IDs stored in the SQLite checkpointer."""
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)
