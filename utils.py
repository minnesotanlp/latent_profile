import concurrent.futures
import csv
import json
import os
import re
from pathlib import Path

import yaml

try:
    import numpy as np
except ImportError:
    np = None


DEFAULT_OUTPUT_ROOT = "/lustre/fs0/scratch/jmooney/latent_profile"
DEFAULT_MODELS_CONFIG = "configs/models.yaml"
DEFAULT_RESULTS_SUBDIR = "results"
THINK_PATTERN = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
PARTIAL_ENDINGS = (
    "i should",
    "my response is",
    "the user",
    "let me",
    "i need to",
    "first, i",
    "so, the response",
)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def require_numpy():
    if np is None:
        raise RuntimeError("numpy is required for this command. Install the project runtime dependencies first.")
    return np


def resolve_output_root(output_root=None):
    root = output_root or os.environ.get("LATENT_PROFILE_OUTPUT_ROOT") or DEFAULT_OUTPUT_ROOT
    ensure_dir(root)
    return root


def build_output_path(output_root, *parts):
    return os.path.join(resolve_output_root(output_root), *parts)


def write_json(path, data):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def append_jsonl(path, rows):
    ensure_dir(Path(path).parent)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_jsonl(path, rows):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_csv(path, fieldnames, rows):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def json_reader(fname, field_path):
    with open(fname, "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in field_path:
        data = data[key]
    return data


def get_topic(topic):
    return json_reader("variations/direct-tests.json", ["topic", topic])


def get_conv_topic(conv_topic):
    return json_reader("variations/direct-tests.json", ["conv_topic", conv_topic])


def get_flex_attributes(topic_correlation, topic):
    if topic_correlation == 0:
        return [None]

    val_0 = json_reader("variations/direct-tests.json", ["topic-agent-prompts", str(topic_correlation), "0", topic])
    val_1 = json_reader("variations/direct-tests.json", ["topic-agent-prompts", str(topic_correlation), "1", topic])
    return [val_0, val_1]


def get_personality_questions(personality_type):
    return json_reader("variations/direct-tests.json", ["personality", str(personality_type)])


def get_prompt_string(fname, prompt_id):
    return json_reader(f"variations/prompts/{fname}.json", [str(prompt_id)])


def load_model_registry(config_path=DEFAULT_MODELS_CONFIG):
    if os.path.exists(config_path):
        data = load_yaml(config_path)
        return data["models"]

    legacy = json_reader("variations/models.json", [])
    models = []
    for model_id, hf_name in legacy.items():
        models.append(
            {
                "id": str(model_id),
                "hf_name": hf_name,
                "family": "legacy",
                "size_b": None,
                "kind": "unknown",
                "served_via": "vllm",
                "supports_chat": True,
                "notes": "Loaded from legacy variations/models.json",
            }
        )
    return models


def get_model_record(model_id, config_path=DEFAULT_MODELS_CONFIG):
    models = load_model_registry(config_path)
    model_id = str(model_id)
    for record in models:
        if str(record["id"]) == model_id or record["hf_name"] == model_id:
            return record
    raise KeyError(f"Unknown model identifier: {model_id}")


def get_model(model_id, port=8000, config_path=DEFAULT_MODELS_CONFIG):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is required to connect to the local vLLM server.") from exc

    record = get_model_record(model_id, config_path)
    client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")
    return client, record["hf_name"], record


def general_system_prompt(demo_list, demo_constructor, sys_string):
    cur_str = demo_constructor(*demo_list)
    cur_str += sys_string
    return cur_str


def system_prompt_alteration(gen_prompt, alteration_prompt):
    return f"{gen_prompt}\n{alteration_prompt}\n"


def build_self_report_messages(sys_prompt, prompt_text):
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt_text},
    ]


def build_agent_conversation_messages(sys_prompt, topic_prompt, turns, speaking_agent):
    messages = [{"role": "system", "content": sys_prompt}]
    messages.append({"role": "user", "content": topic_prompt})
    for turn in turns:
        content = turn.get("public_text", "").strip()
        if not content:
            continue
        role = "assistant" if turn["speaker"] == speaking_agent else "user"
        messages.append({"role": role, "content": content})
    return messages


def sanitize_public_utterance(text, max_chars_hint=1400):
    raw_text = (text or "").strip()
    contained_reasoning = bool(THINK_PATTERN.search(raw_text)) or "<think>" in raw_text.lower()
    lowered_raw = raw_text.lower()
    if "<think>" in lowered_raw and "</think>" not in lowered_raw:
        public_text = ""
    else:
        public_text = THINK_PATTERN.sub("", raw_text).strip()

    # Strip standalone think tags if the model emitted malformed XML-like blocks.
    public_text = re.sub(r"</?think>", "", public_text, flags=re.IGNORECASE).strip()
    public_text = re.sub(r"\n{3,}", "\n\n", public_text)

    lowered = public_text.lower().rstrip()
    was_truncated = False
    if public_text:
        if len(raw_text) >= max_chars_hint and public_text[-1] not in ".!?\"'":
            was_truncated = True
        if lowered.endswith(PARTIAL_ENDINGS):
            was_truncated = True
        if public_text.endswith(":") or public_text.endswith(","):
            was_truncated = True

    ended_with_goodbye = public_text.endswith("Goodbye.") or public_text.endswith("Goodbye")
    return {
        "raw_text": raw_text,
        "public_text": public_text,
        "contained_reasoning": contained_reasoning,
        "was_truncated": was_truncated,
        "ended_with_goodbye": ended_with_goodbye,
    }


def sanitize_choice(text, guided_choices=None):
    value = text.strip()
    if not guided_choices:
        return value

    for choice in guided_choices:
        if value == choice:
            return choice

    match = re.search(r"(-?\d+|Yes|No)", value, flags=re.IGNORECASE)
    if not match:
        return value

    candidate = match.group(1)
    for choice in guided_choices:
        if candidate.lower() == str(choice).lower():
            return str(choice)
    return candidate


def single_generation(
    messages,
    client,
    model_name,
    *,
    guided_choices=None,
    max_tokens=None,
    temperature=None,
    top_p=None,
):
    if guided_choices:
        completion_max_tokens = max_tokens or 4
        completion_temperature = 0.0 if temperature is None else temperature
        completion_top_p = 1.0 if top_p is None else top_p
    else:
        completion_max_tokens = max_tokens or 384
        completion_temperature = 0.5 if temperature is None else temperature
        completion_top_p = 0.9 if top_p is None else top_p

    extra_body = {}
    if guided_choices:
        extra_body["guided_choice"] = [str(choice) for choice in guided_choices]

    request_kwargs = {
        "model": model_name,
        "messages": messages,
        "max_tokens": completion_max_tokens,
        "temperature": completion_temperature,
        "top_p": completion_top_p,
    }
    if extra_body:
        request_kwargs["extra_body"] = extra_body

    response = client.chat.completions.create(**request_kwargs)
    content = response.choices[0].message.content or ""
    return sanitize_choice(content, guided_choices)


def all_generations(
    all_messages,
    client,
    model_name,
    *,
    guided_choices=None,
    max_tokens=None,
    temperature=None,
    top_p=None,
    max_workers=None,
):
    def run(messages):
        return single_generation(
            messages,
            client,
            model_name,
            guided_choices=guided_choices,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(run, all_messages))


def bin_responses(num_bins, arr, max_val):
    require_numpy()
    bins = np.linspace(0, max_val, num_bins + 1)
    bin_ids = np.digitize(arr, bins, right=False) - 1
    return np.clip(bin_ids, 0, num_bins - 1)


def get_stage_dir(output_root, save_dir, model_id, topic, bias):
    return build_output_path(output_root, DEFAULT_RESULTS_SUBDIR, save_dir, str(model_id), str(topic), str(bias))


def write_manifest(stage_dir, stage_name, args_dict, extra=None):
    payload = {
        "stage": stage_name,
        "args": args_dict,
    }
    if extra:
        payload.update(extra)
    write_json(os.path.join(stage_dir, "manifest.json"), payload)
