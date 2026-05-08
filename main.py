import argparse
import itertools
import os

try:
    import numpy as np
except ImportError:
    np = None

from utils import (
    all_generations,
    ensure_dir,
    general_system_prompt,
    get_flex_attributes,
    get_model,
    get_personality_questions,
    get_prompt_string,
    get_stage_dir,
    get_topic,
    require_numpy,
    write_json,
    write_jsonl,
    write_manifest,
    build_self_report_messages,
)
from variations.prompts.demographics import demo_shim


parser = argparse.ArgumentParser(description="Profile agents for topic stance and persuadability.")
parser.add_argument("--topic", type=int, help="Topic index (0-8)")
parser.add_argument("--bias", type=int, help="How correlated the system prompt is with the topic (0-2)")
parser.add_argument("--model-id", type=str, help="Model id from configs/models.yaml or raw HF model name")
parser.add_argument("--personality-type", type=int, default=0, help="Which question battery to use.")
parser.add_argument("--save-dir", type=str, default="core", help="Experiment subdirectory name.")
parser.add_argument("--demo-prompt", type=int, default=0, help="Demographic prompt constructor.")
parser.add_argument("--qa-prompt", type=int, default=0, help="Prompt template for self-report questions.")
parser.add_argument("--topic-qa-prompt", type=int, default=None, help="Prompt template for 1-5 topic responses.")
parser.add_argument("--personality-qa-prompt", type=int, default=None, help="Prompt template for Yes/No persuadability items.")
parser.add_argument("--sys-prompt", type=int, default=0, help="System prompt template.")
parser.add_argument("--port", type=int, default=8000, help="Port of the OpenAI-compatible server.")
parser.add_argument("--preference-respond", type=int, default=1, help="Whether to collect topic preference.")
parser.add_argument("--personality-respond", type=int, default=1, help="Whether to collect persuadability responses.")
parser.add_argument("--one-five", type=int, default=0, help="Backward-compatible no-op kept for legacy scripts.")
parser.add_argument("--output-root", type=str, default=None, help="Root for generated artifacts.")
args = parser.parse_args()
require_numpy()

stage_dir = get_stage_dir(args.output_root, args.save_dir, args.model_id, args.topic, args.bias)
ensure_dir(stage_dir)

demo_constructor, all_attributes = demo_shim(args.demo_prompt)
flexible_attributes = get_flex_attributes(args.bias, args.topic)
all_attributes.append(flexible_attributes)
all_combinations = list(itertools.product(*all_attributes))

client, model_name, model_record = get_model(args.model_id, args.port)
topic_question = get_topic(args.topic)
personality_questions = get_personality_questions(args.personality_type)
topic_qa_prompt = get_prompt_string("qa", args.topic_qa_prompt if args.topic_qa_prompt is not None else args.qa_prompt)
personality_qa_prompt = get_prompt_string("qa", args.personality_qa_prompt if args.personality_qa_prompt is not None else args.qa_prompt)
sys_string = get_prompt_string("gen_sys", args.sys_prompt)

sys_prompts = [general_system_prompt(cur_combination, demo_constructor, sys_string) for cur_combination in all_combinations]

profile_rows = []
topic_responses = []
personality_responses = []

if args.preference_respond:
    topic_messages = [
        build_self_report_messages(sys_prompt, topic_qa_prompt + topic_question)
        for sys_prompt in sys_prompts
    ]
    topic_outputs = all_generations(
        topic_messages,
        client,
        model_name,
        guided_choices=["1", "2", "3", "4", "5"],
        max_tokens=4,
    )
    topic_responses = np.array(topic_outputs, dtype=int)
    np.save(os.path.join(stage_dir, "topic.npy"), topic_responses)

if args.personality_respond:
    all_personality_responses = []
    for cur_question in personality_questions:
        cur_messages = [
            build_self_report_messages(sys_prompt, personality_qa_prompt + cur_question)
            for sys_prompt in sys_prompts
        ]
        cur_outputs = all_generations(
            cur_messages,
            client,
            model_name,
            guided_choices=["Yes", "No"],
            max_tokens=4,
        )
        numeric_outputs = np.array([1 if value == "Yes" else 0 for value in cur_outputs], dtype=int)
        all_personality_responses.append(numeric_outputs)
    personality_responses = np.array(all_personality_responses, dtype=int)
    np.save(os.path.join(stage_dir, "personality.npy"), personality_responses)

for idx, combination in enumerate(all_combinations):
    row = {
        "agent_idx": idx,
        "topic": args.topic,
        "bias": args.bias,
        "model_id": str(args.model_id),
        "model_name": model_name,
        "system_prompt": sys_prompts[idx],
        "demographic_attributes": list(combination[:-1]),
        "topic_bias_prompt": combination[-1],
        "topic_response": int(topic_responses[idx]) if len(topic_responses) else None,
        "persuadability_score": int(np.sum(personality_responses[:, idx])) if len(personality_responses) else None,
        "persuadability_items": personality_responses[:, idx].tolist() if len(personality_responses) else None,
    }
    profile_rows.append(row)

write_jsonl(os.path.join(stage_dir, "profiles.jsonl"), profile_rows)
write_json(
    os.path.join(stage_dir, "args.json"),
    {
        "topic": args.topic,
        "bias": args.bias,
        "model_id": str(args.model_id),
        "model_name": model_name,
        "model_record": model_record,
        "save_dir": args.save_dir,
        "demo_prompt": args.demo_prompt,
        "qa_prompt": args.qa_prompt,
        "topic_qa_prompt": args.topic_qa_prompt,
        "personality_qa_prompt": args.personality_qa_prompt,
        "sys_prompt": args.sys_prompt,
        "personality_type": args.personality_type,
    },
)
np.save(os.path.join(stage_dir, "sys_prompts.npy"), np.array(sys_prompts, dtype=object))
np.save(os.path.join(stage_dir, "all_combinations.npy"), np.array(all_combinations, dtype=object))
write_manifest(
    stage_dir,
    "profile",
    vars(args),
    extra={
        "num_agents": len(profile_rows),
        "topic_responses_written": bool(args.preference_respond),
        "personality_responses_written": bool(args.personality_respond),
    },
)
