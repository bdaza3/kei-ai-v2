import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

import langgraph_agent


class _FakeBoundModel:
    def invoke(self, messages):
        return AIMessage(content="Graph response")


class _FakeModel:
    def bind_tools(self, tools):
        self.tools = tools
        return _FakeBoundModel()


class _ToolCallSequenceModel:
    def __init__(self):
        self.responses = iter([
            AIMessage(content="productivity_agent"),
            AIMessage(
                content="",
                tool_calls=[{"name": "get_current_time", "args": {}, "id": "call-1", "type": "tool_call"}],
            ),
            AIMessage(content="general_agent"),
            AIMessage(content="Tool result received"),
        ])

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return next(self.responses)


class LangGraphAgentTests(unittest.TestCase):
    def test_supervisor_prompt_and_fallback_route(self):
        self.assertIn("executive manager", langgraph_agent.SUPERVISOR_SYSTEM_PROMPT)
        self.assertEqual(
            langgraph_agent._supervisor({"user_text": "check my focus status"})["route"],
            "productivity",
        )

    def test_productivity_worker_uses_role_specific_model(self):
        fake_model = _FakeModel()
        with patch.dict("os.environ", {"KEI_PRODUCTIVITY_MODEL": "openai/gpt-4o-mini"}, clear=False):
            with patch.object(langgraph_agent, "_model", return_value=fake_model) as model_factory:
                langgraph_agent._productivity_agent({"messages": []})
        model_factory.assert_called_once_with("openai/gpt-4o-mini")

    def test_graph_description_exposes_flow_and_tools(self):
        description = langgraph_agent.describe_graph()
        self.assertIn("supervisor", description["nodes"])
        self.assertIn("tools", description["nodes"])
        self.assertEqual(
            description["tools"],
            ["get_active_window", "get_activity_snapshot", "get_current_time"],
        )

    def test_graph_mermaid_contains_tool_loop(self):
        diagram = langgraph_agent.get_graph_mermaid()
        self.assertIn("supervisor", diagram)
        self.assertIn("tools", diagram)

    def test_routes_productivity_requests(self):
        state = {"user_text": "check my focus status"}
        self.assertEqual(langgraph_agent._route_request(state), {"route": "productivity"})

    def test_routes_general_requests(self):
        state = {"user_text": "tell me a short story"}
        self.assertEqual(langgraph_agent._route_request(state), {"route": "general"})

    def test_graph_returns_final_text_without_network(self):
        with patch.object(langgraph_agent, "_model", return_value=_FakeModel()):
            result = langgraph_agent.run_langgraph_agent("hello")
        self.assertEqual(result, "Graph response")

    def test_trace_reports_graph_stages_without_network(self):
        with patch.object(langgraph_agent, "_model", return_value=_FakeModel()):
            result = langgraph_agent.run_langgraph_agent_trace("hello")
        self.assertEqual(result["response"], "Graph response")
        nodes = [event["node"] for event in result["trace"]]
        self.assertEqual(nodes[:2], ["supervisor", "general_agent"])

    def test_trace_reports_tool_call_and_return_to_supervisor(self):
        with patch.object(langgraph_agent, "_model", return_value=_ToolCallSequenceModel()):
            result = langgraph_agent.run_langgraph_agent_trace("check the current time")
        tool_events = [
            message
            for event in result["trace"]
            for message in event.get("messages", [])
            if message.get("type") == "tool_call"
        ]
        nodes = [event["node"] for event in result["trace"]]
        self.assertEqual(tool_events[0]["tools"][0]["name"], "get_current_time")
        self.assertIn("tools", nodes)
        self.assertGreaterEqual(nodes.count("supervisor"), 2)


if __name__ == "__main__":
    unittest.main()
