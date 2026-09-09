import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

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
        tool_events = [event for event in result["trace"] if event.get("decision") == "tool_call"]
        nodes = [event["node"] for event in result["trace"]]
        self.assertEqual(tool_events[0]["tool_calls"][0]["name"], "get_current_time")
        self.assertIn("tools", nodes)
        self.assertGreaterEqual(nodes.count("supervisor"), 2)

    def test_trace_has_explicit_state_metadata_and_end_decision(self):
        with patch.object(langgraph_agent, "_model", return_value=_FakeModel()):
            result = langgraph_agent.run_langgraph_agent_trace("hello")
        self.assertTrue(result["trace"])
        final_event = result["trace"][-1]
        self.assertEqual(final_event["node"], "general_agent")
        self.assertEqual(final_event["decision"], "END")
        self.assertTrue(final_event["task_complete"])

    def test_supervisor_enforces_iteration_limit(self):
        with patch.dict("os.environ", {"KEI_GRAPH_MAX_ITERATIONS": "0"}, clear=False):
            result = langgraph_agent._supervisor({"user_text": "hello", "iteration_count": 0})
        self.assertEqual(result["selected_agent"], "finish")
        self.assertTrue(result["task_complete"])
        self.assertEqual(result["supervisor_decision"], "iteration_limit")

    def test_tool_errors_are_structured_and_complete_the_task(self):
        state = {
            "messages": [
                HumanMessage(content="test"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "missing_tool", "args": {}, "id": "bad-1", "type": "tool_call"}],
                ),
            ],
            "iteration_count": 1,
        }
        result = langgraph_agent._tools(state)
        self.assertTrue(result["task_complete"])
        self.assertEqual(result["trace_events"][0]["reason"], "tool_error")


if __name__ == "__main__":
    unittest.main()
