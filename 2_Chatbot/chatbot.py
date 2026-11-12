from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Iterator, Callable
from langchain_core.messages import BaseMessage, HumanMessage, AIMessageChunk
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", streaming=True)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}


# Checkpointer
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


def stream_chat(
    user_input: str,
    thread_id: str,
    on_step: Callable[[dict], None] | None = None,
) -> Iterator[str]:
    """Stream assistant response tokens for a given user input and thread ID."""
    config = {'configurable': {'thread_id': thread_id}}
    seen_nodes = set()
    for message_chunk, metadata in chatbot.stream(
        {'messages': [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="messages",
    ):
        node = metadata.get("langgraph_node")
        if on_step and node and node not in seen_nodes:
            seen_nodes.add(node)
            on_step(metadata)
        if isinstance(message_chunk, AIMessageChunk) and message_chunk.text:
            yield str(message_chunk.text)



def stream_chat(
    user_input: str,
    thread_id: str,
    on_step: Callable[[dict], None] | None = None,
) -> Iterator[str]:
    """Stream assistant response tokens for a given user input and thread ID."""
    config = {'configurable': {'thread_id': thread_id}}
    seen_nodes = set()
    for message_chunk, metadata in chatbot.stream(
        {'messages': [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="messages",
    ):
        node = metadata.get("langgraph_node")
        if on_step and node and node not in seen_nodes:
            seen_nodes.add(node)
            on_step(metadata)
        if isinstance(message_chunk, AIMessageChunk) and message_chunk.text:
            yield str(message_chunk.text)