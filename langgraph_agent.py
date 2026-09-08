"""LangGraph orchestration for Kei's multi-agent assistant.

The graph is intentionally small but establishes the extension points for more
specialist agents and tools:

    route_request -> specialist agent -> tool executor -> specialist agent
                                      -> final response

The model never executes tools directly. LangGraph routes tool calls through
an allowlisted ToolNode, and each request has a bounded recursion limit.
"""

from __future__ import annotations

import os
import json
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

load_dotenv()

DEFAULT_MODEL = "google/gemma-3-4b-it:free"
MAX_GRAPH_STEPS = 12


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], lambda left, right: left + right]
    route: Literal["productivity", "general"]
    user_text: str
    context: dict[str, object]
    model_name: str | None


def _build_langchain_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            TOOL_FUNCTIONS[schema["function"]["name"]],
            name=schema["function"]["name"],
            description=schema["function"]["description"],
        )
        for schema in TOOL_SCHEMAS
    ]


TOOLS = _build_langchain_tools()


def _model(model_name: str | None = None) -> ChatOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("GEMMA3_4B_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing")

    return ChatOpenAI(
        model=model_name or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL),
        api_key=api_key,
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        temperature=0.2,
        timeout=float(os.environ.get("OPENROUTER_TIMEOUT", "20")),
        max_retries=1,
        default_headers={
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Kei AI",
        },
    )


def _route_request(state: AgentState) -> dict[str, str]:
    """Route deterministically so routing does not add another LLM call."""
    text = state.get("user_text", "").lower()
    productivity_terms = (
        "focus",
        "work",
        "task",
        "pomodoro",
        "distract",
        "procrastinat",
        "status",
        "break",
        "desktop",
        "window",
    )
    route = "productivity" if any(term in text for term in productivity_terms) else "general"
    return {"route": route}


def _specialist_system(route: str) -> str:
    if route == "productivity":
        role = (
            "You are Kei's productivity specialist. Help the user make concrete progress, "
            "and use the active-window tool when current desktop context matters."
        )
    else:
        role = "You are Kei's general conversation specialist. Answer directly and do not invent facts."

    return (
        f"{role} You are one node in a LangGraph agent. "
        "Use tools only when they provide necessary runtime facts. "
        "Never claim a tool action happened unless a tool result confirms it. "
        "Keep the final answer concise and clear."
    )


def _invoke_specialist(state: AgentState, route: Literal["productivity", "general"]) -> dict[str, list[BaseMessage]]:
    messages = state.get("messages", [])
    model = _model(state.get("model_name")).bind_tools(TOOLS)
    response = model.invoke([SystemMessage(content=_specialist_system(route)), *messages])
    return {"messages": [response]}


def _productivity_agent(state: AgentState) -> dict[str, list[BaseMessage]]:
    return _invoke_specialist(state, "productivity")


def _general_agent(state: AgentState) -> dict[str, list[BaseMessage]]:
    return _invoke_specialist(state, "general")


def _route_after_request(state: AgentState) -> str:
    return state.get("route", "general")


def _route_after_tools(state: AgentState) -> str:
    return state.get("route", "general")


def build_agent_graph():
    """Compile the supervisor/specialist/tool graph."""
    graph = StateGraph(AgentState)
    graph.add_node("route_request", _route_request)
    graph.add_node("productivity_agent", _productivity_agent)
    graph.add_node("general_agent", _general_agent)
    graph.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))

    graph.add_edge(START, "route_request")
    graph.add_conditional_edges(
        "route_request",
        _route_after_request,
        {"productivity": "productivity_agent", "general": "general_agent"},
    )

    for agent_name in ("productivity_agent", "general_agent"):
        graph.add_conditional_edges(
            agent_name,
            tools_condition,
            {"tools": "tools", END: END},
        )

    graph.add_conditional_edges(
        "tools",
        _route_after_tools,
        {"productivity": "productivity_agent", "general": "general_agent"},
    )
    return graph.compile()


def describe_graph() -> dict[str, object]:
    """Return the graph topology and registered tools for debugging or docs."""
    graph = get_agent_graph().get_graph()
    return {
        "nodes": sorted(graph.nodes),
        "edges": [{"from": edge.source, "to": edge.target} for edge in graph.edges],
        "tools": [schema["function"]["name"] for schema in TOOL_SCHEMAS],
        "decision_loop": (
            "route_request selects a specialist; the specialist asks the LLM; "
            "tools_condition sends tool calls to tools; tool results return to the specialist; "
            "a text response ends the graph."
        ),
    }


def get_graph_mermaid() -> str:
    """Return a Mermaid diagram of the compiled graph."""
    return get_agent_graph().get_graph().draw_mermaid()


_GRAPH = None


def get_agent_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_agent_graph()
    return _GRAPH


def run_langgraph_agent(
    user_text: str,
    context: dict[str, object] | None = None,
    *,
    model: str | None = None,
) -> str:
    """Run one bounded LangGraph request and return the final assistant text."""
    if not user_text.strip():
        raise ValueError("user_text must not be empty")

    initial: AgentState = {
        "user_text": user_text,
        "context": context or {},
        "model_name": model,
        "messages": [
            HumanMessage(
                content=(
                    f"User request: {user_text}\n\n"
                    f"Runtime context: {json.dumps(context or {}, ensure_ascii=False)}"
                )
            )
        ],
    }
    result = get_agent_graph().invoke(
        initial,
        config={"recursion_limit": int(os.environ.get("KEI_GRAPH_RECURSION_LIMIT", MAX_GRAPH_STEPS))},
    )
    messages = result.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, AIMessage) and isinstance(message.content, str) and message.content.strip():
            return message.content.strip()
    raise RuntimeError("LangGraph completed without a text response")


__all__ = [
    "AgentState",
    "TOOLS",
    "build_agent_graph",
    "describe_graph",
    "get_agent_graph",
    "get_graph_mermaid",
    "run_langgraph_agent",
]
