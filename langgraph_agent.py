"""LangGraph orchestration for Kei's multi-agent assistant.

The graph is intentionally small but establishes the extension points for more
specialist agents and tools:

    supervisor -> specialist agent -> tool executor -> supervisor
                                                    -> final response

The model never executes tools directly. LangGraph routes tool calls through
an allowlisted ToolNode, and each request has a bounded recursion limit.
"""

from __future__ import annotations

import os
import json
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

load_dotenv()

DEFAULT_MODEL = "google/gemma-3-4b-it"
MAX_GRAPH_STEPS = 12
SupervisorRoute = Literal["productivity", "general"]

SUPERVISOR_SYSTEM_PROMPT = """You are the executive manager of Kei's multi-agent productivity assistant team.
Your sole responsibility is to orchestrate tasks by deciding which specialized
agent should run next based on the current conversation and state.

Available worker agents:
1. 'productivity_agent': Focus, desktop activity, Pomodoro, and concrete work progress.
2. 'general_agent': General conversation and requests that do not need a specialist.

Future worker agents will include calendar_agent, email_agent, and file_agent.
Do not select a future worker until it is registered in the graph.

Rules:
- Analyze the user's latest request and the conversation history.
- For a multi-step task, choose the first necessary registered worker.
- If a worker has just returned information, decide whether another registered worker is needed.
- Choose exactly one registered worker name and output nothing else.
- Never perform the task yourself and never call tools.
"""


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], lambda left, right: left + right]
    route: Literal["productivity", "general"]
    user_text: str
    context: dict[str, object]
    model_name: str | None
    supervisor_model_name: str | None
    supervisor_decision: str


#function to build the list of available tools from the TOOL_FUNCTIONS and TOOL_SCHEMAS
def _build_langchain_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            TOOL_FUNCTIONS[schema["function"]["name"]],
            name=schema["function"]["name"],
            description=schema["function"]["description"],
        )
        for schema in TOOL_SCHEMAS
    ]

# Build the list of available tools
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

#function to provide a fallback routing decision if the supervisor model is unavailable
def _fallback_route_request(state: AgentState) -> SupervisorRoute:
    """Keep routing available if the supervisor model is unavailable."""
    text = state.get("user_text", "").lower()
    productivity_terms = (#list of terms that indicate a request is related to productivity or focus
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
    return "productivity" if any(term in text for term in productivity_terms) else "general"


def _route_request(state: AgentState) -> dict[str, str]:
    """Backward-compatible deterministic routing helper."""
    return {"route": _fallback_route_request(state)}


#function to parse the supervisor's decision and determine the next route in the graph
#e.g., if the supervisor returns "productivity_agent", this function will return "productivity"
def _parse_supervisor_route(content: object) -> SupervisorRoute | None:
    if not isinstance(content, str):
        return None
    candidate = content.strip().lower().strip("` .\n")
    if candidate in {"productivity_agent", "productivity"}:
        return "productivity"
    if candidate in {"general_agent", "general"}:
        return "general"
    return None

#function to handle the supervisor's decision and route the request to the appropriate specialist agent
def _supervisor(state: AgentState) -> dict[str, str]:
    """Ask a separate model to choose the next registered worker."""
    try:
        supervisor_model = state.get("supervisor_model_name") or os.environ.get("KEI_SUPERVISOR_MODEL")
        model = _model(supervisor_model).bind_tools([])
        response = model.invoke([SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT), *state.get("messages", [])])
        route = _parse_supervisor_route(response.content)
        if route is not None:
            return {"route": route, "supervisor_decision": response.content}
    except Exception:
        pass

    fallback = _fallback_route_request(state)
    return {"route": fallback, "supervisor_decision": f"fallback:{fallback}"}

#function to generate the system prompt for the specialist agent based on the route
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
    configured_model = state.get("model_name") or os.environ.get(f"KEI_{route.upper()}_MODEL")
    model = _model(configured_model).bind_tools(TOOLS)
    response = model.invoke([SystemMessage(content=_specialist_system(route)), *messages])
    return {"messages": [response]}


def _productivity_agent(state: AgentState) -> dict[str, list[BaseMessage]]:
    return _invoke_specialist(state, "productivity")


def _general_agent(state: AgentState) -> dict[str, list[BaseMessage]]:
    return _invoke_specialist(state, "general")


def _route_after_request(state: AgentState) -> str:
    return state.get("route", "general")


def build_agent_graph():
    """Compile the supervisor/specialist/tool graph."""
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", _supervisor)
    graph.add_node("productivity_agent", _productivity_agent)
    graph.add_node("general_agent", _general_agent)
    graph.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
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
        lambda _state: "supervisor",
        {"supervisor": "supervisor"},
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
            "supervisor selects a registered specialist; the specialist asks the worker LLM; "
            "tools_condition sends tool calls to tools; tool results return to the supervisor; "
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

#function to run one bounded LangGraph request and return the final assistant text
def run_langgraph_agent(
    user_text: str,
    context: dict[str, object] | None = None,
    *,
    model: str | None = None,
) -> str:
    """Run one bounded LangGraph request and return the final assistant text."""
    return run_langgraph_agent_trace(user_text, context, model=model)["response"]


def _initial_state(
    user_text: str,
    context: dict[str, object] | None,
    model: str | None,
) -> AgentState:
    return {
        "user_text": user_text,
        "context": context or {},
        "model_name": model,
        "supervisor_model_name": os.environ.get("KEI_SUPERVISOR_MODEL"),
        "messages": [
            HumanMessage(
                content=(
                    f"User request: {user_text}\n\n"
                    f"Runtime context: {json.dumps(context or {}, ensure_ascii=False)}"
                )
            )
        ],
    }


def _trace_message(message: BaseMessage) -> dict[str, object] | None:
    if isinstance(message, AIMessage) and message.tool_calls:
        return {
            "type": "tool_call",
            "tools": [
                {"name": call.get("name"), "arguments": call.get("args", {})}
                for call in message.tool_calls
            ],
        }
    if isinstance(message, ToolMessage):
        return {
            "type": "tool_result",
            "name": message.name,
            "result": message.content,
        }
    return None


def run_langgraph_agent_trace(
    user_text: str,
    context: dict[str, object] | None = None,
    *,
    model: str | None = None,
) -> dict[str, object]:
    """Run LangGraph and return the final response plus a safe execution trace."""
    if not user_text.strip():
        raise ValueError("user_text must not be empty")

    messages: list[BaseMessage] = []
    trace: list[dict[str, object]] = []
    initial = _initial_state(user_text, context, model)
    for update in get_agent_graph().stream(
        initial,
        config={"recursion_limit": int(os.environ.get("KEI_GRAPH_RECURSION_LIMIT", MAX_GRAPH_STEPS))},
        stream_mode="updates",
    ):
        for node, delta in update.items():
            event: dict[str, object] = {"node": node}
            if isinstance(delta, dict):
                for field in ("route", "supervisor_decision"):
                    if field in delta:
                        event[field] = delta[field]
                new_messages = delta.get("messages", [])
                if isinstance(new_messages, list):
                    messages.extend(new_messages)
                    message_events = [
                        message_event
                        for message in new_messages
                        if (message_event := _trace_message(message)) is not None
                    ]
                    if message_events:
                        event["messages"] = message_events
            trace.append(event)

    for message in reversed(messages):
        if isinstance(message, AIMessage) and isinstance(message.content, str) and message.content.strip():
            return {
                "response": message.content.strip(),
                "trace": trace,
                "models": {
                    "supervisor": os.environ.get("KEI_SUPERVISOR_MODEL"),
                    "productivity": os.environ.get("KEI_PRODUCTIVITY_MODEL"),
                    "general": os.environ.get("KEI_GENERAL_MODEL"),
                },
            }
    raise RuntimeError("LangGraph completed without a text response")


__all__ = [
    "AgentState",
    "TOOLS",
    "build_agent_graph",
    "describe_graph",
    "get_agent_graph",
    "get_graph_mermaid",
    "run_langgraph_agent",
    "run_langgraph_agent_trace",
]
