from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from smart_home_langgraph.graph.workflow import build_workflow
from smart_home_langgraph.main import ChatSession


def test_chat_session_keeps_history_between_turns():
    captured_states: list[dict] = []

    def fake_generator(state):
        captured_states.append(state)
        return f"echo: {state['user_query']}", False

    def fake_critique(state):
        return {"passed": True, "issues": [], "severity": "low", "repair_hints": ""}

    app = build_workflow(
        response_generator=fake_generator,
        critique_generator=fake_critique,
        checkpointer=MemorySaver(),
    )
    session = ChatSession(app=app)

    first_response, _ = session.ask("How can I save energy?")
    second_response, _ = session.ask("What should I try first?")

    assert first_response == "echo: How can I save energy?"
    assert second_response == "echo: What should I try first?"
    assert captured_states[0]["conversation_history"] == []
    assert len(captured_states[1]["conversation_history"]) == 2
    assert captured_states[1]["conversation_history"][0] == HumanMessage(
        content="How can I save energy?"
    )
    assert captured_states[1]["conversation_history"][1] == AIMessage(
        content="echo: How can I save energy?"
    )


def test_chat_session_reset_clears_history():
    captured_states: list[dict] = []

    def fake_generator(state):
        captured_states.append(state)
        return "ok", False

    def fake_critique(state):
        return {"passed": True, "issues": [], "severity": "low", "repair_hints": ""}

    session = ChatSession(
        app=build_workflow(
            response_generator=fake_generator,
            critique_generator=fake_critique,
            checkpointer=MemorySaver(),
        )
    )

    session.ask("Hello")
    session.ask("Again")
    assert len(captured_states[1]["conversation_history"]) == 2

    session.reset()

    session.ask("Fresh start")

    assert captured_states[2]["conversation_history"] == []
