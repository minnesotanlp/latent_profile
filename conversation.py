import argparse
import json
import os

try:
    import numpy as np
except ImportError:
    np = None

from utils import (
    all_generations,
    bin_responses,
    build_agent_conversation_messages,
    ensure_dir,
    get_conv_topic,
    get_model,
    get_prompt_string,
    get_stage_dir,
    require_numpy,
    sanitize_public_utterance,
    system_prompt_alteration,
    write_json,
    write_jsonl,
    write_manifest,
)


parser = argparse.ArgumentParser(description="Sample agent pairs and generate conversations.")
parser.add_argument("--topic", type=int, help="Topic index (0-8)")
parser.add_argument("--bias1", type=int, help="Bias level for the first agent pool.")
parser.add_argument("--bias2", type=int, help="Bias level for the second agent pool.")
parser.add_argument("--model-id", type=str, help="Model id from configs/models.yaml or raw HF model name")
parser.add_argument("--save-dir", type=str, default="core", help="Experiment subdirectory name.")
parser.add_argument("--port", type=int, default=8000, help="Port of the OpenAI-compatible server.")
parser.add_argument("--personality-bins", type=int, default=9, help="Number of persuadability bins.")
parser.add_argument("--topic-bins", type=int, default=6, help="Number of topic preference bins.")
parser.add_argument("--conv-threshold", type=int, default=10, help="Minimum support for a sampled bin.")
parser.add_argument("--sample-size", type=int, default=1, help="Number of conversations per bin pair.")
parser.add_argument("--conv-prompt", type=int, default=0, help="Conversation prompt template id.")
parser.add_argument("--num-rounds", type=int, default=2, help="Number of turns taken by each agent.")
parser.add_argument("--seed", type=int, default=0, help="Random seed for pairing.")
parser.add_argument("--profile-source-dir", type=str, default=None, help="Experiment subdirectory to read profile outputs from.")
parser.add_argument("--output-root", type=str, default=None, help="Root for generated artifacts.")
args = parser.parse_args()
require_numpy()


def get_bias_data(bias):
    profile_source_dir = args.profile_source_dir or args.save_dir
    stage_dir = get_stage_dir(args.output_root, profile_source_dir, args.model_id, args.topic, bias)
    sys_prompts = np.load(os.path.join(stage_dir, "sys_prompts.npy"), allow_pickle=True)
    topic_data = np.load(os.path.join(stage_dir, "topic.npy"), allow_pickle=True).astype(int)
    personality_data = np.load(os.path.join(stage_dir, "personality.npy"), allow_pickle=True).astype(int)

    conv_prompt = get_prompt_string("conversation", args.conv_prompt)
    conversation_system_prompts = [system_prompt_alteration(gen_prompt, conv_prompt) for gen_prompt in sys_prompts]

    personality_agg = np.sum(personality_data, axis=0)
    topic_ids = bin_responses(args.topic_bins, topic_data, 5)
    personality_ids = bin_responses(args.personality_bins, personality_agg, personality_data.shape[0])

    combined_attributes = np.stack((topic_ids, personality_ids), axis=1)
    unique_attributes, inverse_indices, counts = np.unique(
        combined_attributes,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    return conversation_system_prompts, topic_data, personality_agg, unique_attributes, inverse_indices, counts


topic_discussion = get_conv_topic(args.topic)
client, model_name, _ = get_model(args.model_id, args.port)
rng = np.random.default_rng(args.seed)

c_prompts1, topic_data1, personality_agg1, u_atts1, inv_indices1, counts1 = get_bias_data(args.bias1)
c_prompts2, topic_data2, personality_agg2, u_atts2, inv_indices2, counts2 = get_bias_data(args.bias2)

chosen_indices = np.zeros((len(u_atts1), len(u_atts2), 2, args.sample_size), dtype=int)
conversations = np.full(
    (len(u_atts1), len(u_atts2), args.sample_size, 1 + args.num_rounds * 2),
    "",
    dtype=object,
)
conversation_rows = []

for idx0, _ in enumerate(u_atts1):
    for idx1, _ in enumerate(u_atts2):
        if counts1[idx0] < args.conv_threshold or counts2[idx1] < args.conv_threshold:
            continue

        agent0_indices = np.where(idx0 == inv_indices1)[0]
        agent1_indices = np.where(idx1 == inv_indices2)[0]
        agent0_sample = rng.choice(agent0_indices, size=args.sample_size, replace=True)
        agent1_sample = rng.choice(agent1_indices, size=args.sample_size, replace=True)

        chosen_indices[idx0, idx1, 0] = agent0_sample
        chosen_indices[idx0, idx1, 1] = agent1_sample
        all_turns = [[] for _ in range(args.sample_size)]

        for _round_idx in range(args.num_rounds):
            agent0_messages = [
                build_agent_conversation_messages(
                    c_prompts1[agent_idx],
                    topic_discussion,
                    all_turns[conv_idx],
                    "agent_1",
                )
                for conv_idx, agent_idx in enumerate(agent0_sample)
            ]
            agent0_outputs = all_generations(agent0_messages, client, model_name)
            for sample_idx in range(args.sample_size):
                cleaned_turn = sanitize_public_utterance(agent0_outputs[sample_idx])
                cleaned_turn["speaker"] = "agent_1"
                all_turns[sample_idx].append(cleaned_turn)

            agent1_messages = [
                build_agent_conversation_messages(
                    c_prompts2[agent_idx],
                    topic_discussion,
                    all_turns[conv_idx],
                    "agent_2",
                )
                for conv_idx, agent_idx in enumerate(agent1_sample)
            ]
            agent1_outputs = all_generations(agent1_messages, client, model_name)
            for sample_idx in range(args.sample_size):
                cleaned_turn = sanitize_public_utterance(agent1_outputs[sample_idx])
                cleaned_turn["speaker"] = "agent_2"
                all_turns[sample_idx].append(cleaned_turn)

        public_conversations = []
        for sample_idx, turns in enumerate(all_turns):
            public_utterances = [turn["public_text"] for turn in turns]
            num_reasoning_leaks = sum(int(turn["contained_reasoning"]) for turn in turns)
            num_truncated_turns = sum(int(turn["was_truncated"]) for turn in turns)
            num_empty_public_turns = sum(int(not turn["public_text"].strip()) for turn in turns)
            repeated_public_turns = int(len(public_utterances) >= 2 and len(set(public_utterances)) == 1)
            quality_flag = any(
                [
                    num_empty_public_turns > 0,
                    repeated_public_turns > 0,
                    num_truncated_turns > 0,
                ]
            )

            row = {
                "pair_bin_1": int(idx0),
                "pair_bin_2": int(idx1),
                "sample_idx": int(sample_idx),
                "agent_1_idx": int(agent0_sample[sample_idx]),
                "agent_2_idx": int(agent1_sample[sample_idx]),
                "agent_1_topic_response": int(topic_data1[agent0_sample[sample_idx]]),
                "agent_2_topic_response": int(topic_data2[agent1_sample[sample_idx]]),
                "agent_1_persuadability": int(personality_agg1[agent0_sample[sample_idx]]),
                "agent_2_persuadability": int(personality_agg2[agent1_sample[sample_idx]]),
                "topic_prompt": topic_discussion,
                "turns": turns,
                "utterances": public_utterances,
                "num_reasoning_leaks": int(num_reasoning_leaks),
                "num_truncated_turns": int(num_truncated_turns),
                "num_empty_public_turns": int(num_empty_public_turns),
                "repeated_public_turns": repeated_public_turns,
                "conversation_quality_flag": bool(quality_flag),
            }
            conversation_rows.append(row)
            public_conversations.append([topic_discussion] + public_utterances)

        public_conversations = np.array(public_conversations, dtype=object)
        conversations[idx0][idx1] = public_conversations

bias_location = 3 * args.bias1 + args.bias2
final_stage_dir = get_stage_dir(args.output_root, args.save_dir, args.model_id, args.topic, bias_location)
ensure_dir(final_stage_dir)

np.save(os.path.join(final_stage_dir, "conversations.npy"), conversations, allow_pickle=True)
np.save(os.path.join(final_stage_dir, "chosen_indices.npy"), chosen_indices, allow_pickle=True)
np.save(os.path.join(final_stage_dir, "unique_attributes1.npy"), u_atts1, allow_pickle=True)
np.save(os.path.join(final_stage_dir, "unique_attributes2.npy"), u_atts2, allow_pickle=True)
write_jsonl(os.path.join(final_stage_dir, "conversations.jsonl"), conversation_rows)
write_json(
    os.path.join(final_stage_dir, "conv_args.json"),
    {
        "topic": args.topic,
        "bias1": args.bias1,
        "bias2": args.bias2,
        "model_id": str(args.model_id),
        "model_name": model_name,
        "num_rounds": args.num_rounds,
        "sample_size": args.sample_size,
        "seed": args.seed,
    },
)
write_manifest(
    final_stage_dir,
    "conversation",
    vars(args),
    extra={"num_conversations": len(conversation_rows), "combined_bias": bias_location},
)
