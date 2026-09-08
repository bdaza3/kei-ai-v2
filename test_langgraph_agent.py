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


class LangGraphAgentTests(unittest.TestCase):
    def test_supervisor_prompt_and_fallback_route(self):
        self.assertIn("executive manager", langgraph_agent.SUPERVISOR_SYSTEM_PROMPT)
        self.assertEqual(
            langgraph_agent._supervisor({"user_text": "check my focus status"})["route"],
            "productivity",
        )

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


if __name__ == "__main__":
    unittest.main()
