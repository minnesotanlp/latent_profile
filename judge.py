import argparse
import os
import sys

try:
    import numpy as np
except ImportError:
    np = None

from utils import all_generations, get_model, get_stage_dir, require_numpy, write_json, write_manifest
from variations.prompts.judge import judge_shim


parser = argparse.ArgumentParser(description="Judge agent conversations for agreement.")
parser.add_argument("--topic", type=int, help="Topic index (0-8)")
parser.add_argument("--combined-bias", type=int, help="Combined bias location in [0, 8]")
parser.add_argument("--model-id", type=str, help="Agent model id")
parser.add_argument("--save-dir", type=str, default="core", help="Experiment subdirectory name.")
parser.add_argument("--port", type=int, default=8000, help="Port of the OpenAI-compatible server.")
parser.add_argument("--judge-window", type=int, default=3, help="Number of back-and-forth turns shown to judge.")
parser.add_argument("--judge-prompt", type=int, default=0, help="Judge prompt template id.")
parser.add_argument("--judge-model", type=str, default="5", help="Judge model id from configs/models.yaml.")
parser.add_argument("--output-root", type=str, default=None, help="Root for generated artifacts.")
args = parser.parse_args()
require_numpy()


def get_data(stage_dir):
    fname = os.path.join(stage_dir, "conversations.npy")
    if not os.path.exists(fname):
        sys.exit(0)
    return np.load(fname, allow_pickle=True)


def construct_judge_messages(single_conversation, judge_window):
    prepared = []
    for idx, utterance in enumerate(single_conversation):
        if not utterance:
            break
        cleaned = utterance.replace("Goodbye.", "").replace("Goodbye", "").strip()
        if idx == 0:
            prepared.append(cleaned)
            continue
        speaker = "Agent1" if idx % 2 == 1 else "Agent2"
        prepared.append(f"{speaker}: {cleaned}")

    if len(prepared) <= 1:
        return []

    public_turns = [item.split(": ", 1)[-1] for item in prepared[1:]]
    if not any(public_turns):
        return []
    if len(public_turns) >= 2 and len(set(public_turns)) == 1:
        return []

    windows = []
    for idx in range(1, len(prepared)):
        start = max(1, idx - judge_window * 2 + 1)
        content = "\n".join([prepared[0]] + prepared[start : idx + 1]).strip()
        windows.append(content if content else " ")
    return windows


def construct_judge_prompts(judge_sys_prompt, single_conversation, judge_window):
    full_prompts = []
    for cur_content in construct_judge_messages(single_conversation, judge_window):
        full_prompts.append(
            [
                {"role": "system", "content": judge_sys_prompt},
                {"role": "user", "content": cur_content},
            ]
        )
    return full_prompts


stage_dir = get_stage_dir(args.output_root, args.save_dir, args.model_id, args.topic, args.combined_bias)
conversations = get_data(stage_dir)
judge_sys_prompt = judge_shim(args.judge_prompt, args.judge_window)
client, judge_model_name, judge_record = get_model(args.judge_model, args.port)

judge_scores = np.zeros(
    (
        conversations.shape[0],
        conversations.shape[1],
        conversations.shape[2],
        conversations.shape[3] - 1,
    ),
    dtype=int,
) - 1

for idx0 in range(conversations.shape[0]):
    for idx1 in range(conversations.shape[1]):
        combination_prompts = []
        prompt_lengths = []
        for idx2 in range(conversations.shape[2]):
            single_conversation = conversations[idx0][idx1][idx2]
            prompts = construct_judge_prompts(judge_sys_prompt, single_conversation, args.judge_window)
            combination_prompts.extend(prompts)
            prompt_lengths.append(len(prompts))

        if not combination_prompts:
            continue

        cur_scores = all_generations(
            combination_prompts,
            client,
            judge_model_name,
            guided_choices=["1", "2", "3", "4", "5", "-1"],
            max_tokens=4,
        )

        cursor = 0
        for idx2, prompt_length in enumerate(prompt_lengths):
            if prompt_length == 0:
                continue
            parsed = [int(score) for score in cur_scores[cursor : cursor + prompt_length]]
            judge_scores[idx0, idx1, idx2, :prompt_length] = parsed
            cursor += prompt_length

write_json(
    os.path.join(stage_dir, "judge_args.json"),
    {
        "topic": args.topic,
        "combined_bias": args.combined_bias,
        "model_id": str(args.model_id),
        "judge_model": str(args.judge_model),
        "judge_model_name": judge_model_name,
        "judge_model_record": judge_record,
        "judge_window": args.judge_window,
    },
)
np.save(os.path.join(stage_dir, "judge_scores.npy"), judge_scores, allow_pickle=True)
write_manifest(
    stage_dir,
    "judge",
    vars(args),
    extra={"judge_model_name": judge_model_name, "shape": list(judge_scores.shape)},
)
