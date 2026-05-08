import itertools

from utils import (
    build_agent_conversation_messages,
    get_flex_attributes,
    sanitize_public_utterance,
)
from variations.prompts.demographics import demo_shim
from variations.prompts.judge import judge_shim


def test_build_agent_conversation_messages_preserves_public_history():
    turns = [
        {"speaker": "agent_1", "public_text": "I think taxes fund public goods."},
        {"speaker": "agent_2", "public_text": "I worry they can burden households."},
        {"speaker": "agent_1", "public_text": ""},
    ]

    messages = build_agent_conversation_messages(
        "You are agent 1.",
        "Do you think taxes help society?",
        turns,
        "agent_1",
    )

    assert messages == [
        {"role": "system", "content": "You are agent 1."},
        {"role": "user", "content": "Do you think taxes help society?"},
        {"role": "assistant", "content": "I think taxes fund public goods."},
        {"role": "user", "content": "I worry they can burden households."},
    ]


def test_demographic_and_bias_cardinality_matches_paper_design():
    _constructor, attributes = demo_shim(0)
    demographic_count = len(list(itertools.product(*attributes)))

    assert demographic_count == 4 * 5 * 2 * 4 * 6
    assert len(get_flex_attributes(0, 0)) == 1
    assert len(get_flex_attributes(1, 0)) == 2
    assert len(get_flex_attributes(2, 0)) == 2
    assert demographic_count * (1 + 2 + 2) == 4800


def test_sanitize_public_utterance_removes_reasoning_block():
    cleaned = sanitize_public_utterance("<think>hidden</think>Public answer.")

    assert cleaned["public_text"] == "Public answer."
    assert cleaned["contained_reasoning"] is True
    assert cleaned["was_truncated"] is False


def test_judge_prompt_includes_window_with_sentence_boundary():
    prompt = judge_shim(0, 3)

    assert "last 3 statements from each agent." in prompt
    for score in range(1, 6):
        assert f"Score {score}:" in prompt
