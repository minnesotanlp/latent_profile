import argparse
import os
import signal
import socket
import subprocess
import sys
import time

import requests

from utils import (
    build_output_path,
    ensure_dir,
    get_model_record,
    load_yaml,
    resolve_output_root,
    write_json,
)


def wait_for_server(port, timeout_s, interval_s):
    url = f"http://localhost:{port}/v1/models"
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(interval_s)
    raise TimeoutError(f"Timed out waiting for server on port {port}")


def pick_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.getsockname()[1]


def build_runtime_env(output_root):
    env = os.environ.copy()
    cache_root = build_output_path(output_root, "cache")
    tmp_root = build_output_path(output_root, "tmp")
    ensure_dir(cache_root)
    ensure_dir(tmp_root)
    env["HF_HOME"] = cache_root
    env["HUGGINGFACE_HUB_CACHE"] = cache_root
    env["TRANSFORMERS_CACHE"] = cache_root
    env["TMPDIR"] = tmp_root
    env["LATENT_PROFILE_OUTPUT_ROOT"] = output_root
    return env


def start_server(serving_cfg, model_record, env, port):
    cmd = [
        serving_cfg.get("command", "vllm"),
        "serve",
        model_record["hf_name"],
        "--port",
        str(port),
        "--max-model-len",
        str(serving_cfg.get("max_model_len", 4096)),
        "--gpu-memory-utilization",
        str(serving_cfg.get("gpu_memory_utilization", 0.9)),
        "--tensor-parallel-size",
        str(serving_cfg.get("tensor_parallel_size", 1)),
    ]

    cmd.extend(list(serving_cfg.get("extra_args", [])))
    return subprocess.Popen(cmd, preexec_fn=os.setsid, env=env)


def stop_server(proc):
    if proc is None:
        return
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


def run_cmd(cmd, env=None):
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def profile_outputs_exist(root, experiment_name, model_id, topic, bias):
    path = build_output_path(root, "results", experiment_name, str(model_id), str(topic), str(bias), "profiles.jsonl")
    return os.path.exists(path)


def conversation_outputs_exist(root, experiment_name, model_id, topic, combined_bias):
    path = build_output_path(root, "results", experiment_name, str(model_id), str(topic), str(combined_bias), "conversations.npy")
    return os.path.exists(path)


def judge_outputs_exist(root, experiment_name, model_id, topic, combined_bias):
    path = build_output_path(root, "results", experiment_name, str(model_id), str(topic), str(combined_bias), "judge_scores.npy")
    return os.path.exists(path)


def resolve_effective_port(serving_cfg, port_override):
    if port_override is not None:
        return int(port_override)
    env_port = os.environ.get("PIPELINE_PORT") or os.environ.get("VLLM_PORT")
    if env_port:
        return int(env_port)
    return int(serving_cfg["port"])


def start_server_with_retry(serving_cfg, model_record, env, preferred_port):
    max_attempts = int(serving_cfg.get("port_retry_attempts", 5))
    last_error = None
    current_port = int(preferred_port)

    for attempt_idx in range(max_attempts):
        proc = None
        try:
            env["PIPELINE_PORT"] = str(current_port)
            env["VLLM_PORT"] = str(current_port)
            proc = start_server(serving_cfg, model_record, env, current_port)
            wait_for_server(
                current_port,
                serving_cfg.get("max_start_wait_s", 600),
                serving_cfg.get("poll_interval_s", 5),
            )
            return proc, current_port
        except Exception as exc:
            last_error = exc
            stop_server(proc)
            current_port = pick_free_port()

    raise RuntimeError(f"Failed to start vLLM for {model_record['hf_name']} after {max_attempts} attempts") from last_error


def run_pipeline(config_path, resume=False, analyze_only=False, model_ids_override=None, skip_analysis=False, port_override=None):
    config = load_yaml(config_path)
    experiment = config["experiment"]
    serving_cfg = config["serving"]
    stages_cfg = config["stages"]
    if model_ids_override:
        experiment["model_ids"] = [str(model_id) for model_id in model_ids_override]
    profile_source_name = experiment.get("profile_source_name", experiment["name"])
    output_root = resolve_output_root(experiment.get("output_root"))
    runtime_env = build_runtime_env(output_root)
    effective_port = resolve_effective_port(serving_cfg, port_override)
    runtime_env["PIPELINE_PORT"] = str(effective_port)
    runtime_env["VLLM_PORT"] = str(effective_port)
    ensure_dir(build_output_path(output_root, "logs"))
    ensure_dir(build_output_path(output_root, "tables"))
    ensure_dir(build_output_path(output_root, "figures"))

    manifest = {
        "config_path": config_path,
        "experiment": experiment,
        "serving": serving_cfg,
        "effective_port": effective_port,
        "model_ids_override": model_ids_override,
        "skip_analysis": skip_analysis,
        "profile_source_name": profile_source_name,
    }
    write_json(build_output_path(output_root, "logs", f"{experiment['name']}_pipeline_manifest.json"), manifest)

    if analyze_only:
        run_cmd(
            [
                sys.executable,
                "analysis.py",
                "--output-root",
                output_root,
                "--experiment-name",
                experiment["name"],
                "--analysis-name",
                f"{experiment['name']}_summary",
            ],
            env=runtime_env,
        )
        return

    for model_id in experiment["model_ids"]:
        model_record = get_model_record(model_id)
        proc = None
        model_port = effective_port
        try:
            if serving_cfg.get("enabled", True):
                proc, model_port = start_server_with_retry(serving_cfg, model_record, runtime_env, model_port)

            if profile_source_name == experiment["name"]:
                for topic in experiment["topics"]:
                    for bias in experiment["profile_biases"]:
                        if resume and profile_outputs_exist(output_root, experiment["name"], model_id, topic, bias):
                            continue

                        run_cmd(
                            [
                                sys.executable,
                                "main.py",
                                "--topic",
                                str(topic),
                                "--bias",
                                str(bias),
                                "--model-id",
                                str(model_id),
                                "--save-dir",
                                experiment["name"],
                                "--port",
                                str(model_port),
                                "--output-root",
                                output_root,
                                "--demo-prompt",
                                str(stages_cfg["profile"]["demo_prompt"]),
                                "--sys-prompt",
                                str(stages_cfg["profile"]["sys_prompt"]),
                                "--personality-type",
                                str(stages_cfg["profile"]["personality_type"]),
                                "--topic-qa-prompt",
                                str(stages_cfg["profile"]["preference_qa_prompt"]),
                                "--personality-qa-prompt",
                                str(stages_cfg["profile"]["persuadability_qa_prompt"]),
                                "--preference-respond",
                                "1",
                                "--personality-respond",
                                "1",
                            ],
                            env=runtime_env,
                        )

            for topic in experiment["topics"]:
                for bias1 in experiment["profile_biases"]:
                    for bias2 in experiment["profile_biases"]:
                        combined_bias = 3 * bias1 + bias2
                        if resume and conversation_outputs_exist(output_root, experiment["name"], model_id, topic, combined_bias):
                            continue
                        run_cmd(
                            [
                                sys.executable,
                                "conversation.py",
                                "--topic",
                                str(topic),
                                "--bias1",
                                str(bias1),
                                "--bias2",
                                str(bias2),
                                "--model-id",
                                str(model_id),
                                "--save-dir",
                                experiment["name"],
                                "--port",
                                str(model_port),
                                "--output-root",
                                output_root,
                                "--sample-size",
                                str(stages_cfg["conversation"]["sample_size"]),
                                "--conv-threshold",
                                str(stages_cfg["conversation"]["conv_threshold"]),
                                "--topic-bins",
                                str(stages_cfg["conversation"]["topic_bins"]),
                                "--personality-bins",
                                str(stages_cfg["conversation"]["personality_bins"]),
                                "--conv-prompt",
                                str(stages_cfg["conversation"]["conv_prompt"]),
                                "--num-rounds",
                                str(stages_cfg["conversation"]["num_rounds"]),
                                "--seed",
                                str(experiment.get("seed", 0)),
                                "--profile-source-dir",
                                profile_source_name,
                            ],
                            env=runtime_env,
                        )
        finally:
            stop_server(proc)

        judge_proc = None
        judge_port = effective_port
        try:
            judge_model_record = get_model_record(experiment["judge_model_id"])
            if serving_cfg.get("enabled", True):
                judge_proc, judge_port = start_server_with_retry(serving_cfg, judge_model_record, runtime_env, judge_port)

            for topic in experiment["topics"]:
                for combined_bias in experiment["combined_biases"]:
                    if resume and judge_outputs_exist(output_root, experiment["name"], model_id, topic, combined_bias):
                        continue
                    run_cmd(
                        [
                            sys.executable,
                            "judge.py",
                            "--topic",
                            str(topic),
                            "--combined-bias",
                            str(combined_bias),
                            "--model-id",
                            str(model_id),
                            "--save-dir",
                            experiment["name"],
                            "--port",
                            str(judge_port),
                            "--judge-model",
                            str(experiment["judge_model_id"]),
                            "--judge-prompt",
                            str(stages_cfg["judge"]["judge_prompt"]),
                            "--judge-window",
                            str(stages_cfg["judge"]["judge_window"]),
                            "--output-root",
                            output_root,
                        ],
                        env=runtime_env,
                    )
        finally:
            stop_server(judge_proc)

    if not skip_analysis:
        run_cmd(
            [
                sys.executable,
                "analysis.py",
                "--output-root",
                output_root,
                "--experiment-name",
                experiment["name"],
                "--analysis-name",
                f"{experiment['name']}_summary",
            ],
            env=runtime_env,
        )


def main():
    parser = argparse.ArgumentParser(description="Unified experiment pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--model-ids", nargs="*")
    run_parser.add_argument("--skip-analysis", action="store_true")
    run_parser.add_argument("--port", type=int, default=None)

    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("--config", required=True)
    resume_parser.add_argument("--model-ids", nargs="*")
    resume_parser.add_argument("--skip-analysis", action="store_true")
    resume_parser.add_argument("--port", type=int, default=None)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--config", required=True)
    analyze_parser.add_argument("--model-ids", nargs="*")
    analyze_parser.add_argument("--port", type=int, default=None)

    args = parser.parse_args()

    if args.command == "run":
        run_pipeline(
            args.config,
            resume=False,
            analyze_only=False,
            model_ids_override=args.model_ids,
            skip_analysis=args.skip_analysis,
            port_override=args.port,
        )
    elif args.command == "resume":
        run_pipeline(
            args.config,
            resume=True,
            analyze_only=False,
            model_ids_override=args.model_ids,
            skip_analysis=args.skip_analysis,
            port_override=args.port,
        )
    elif args.command == "analyze":
        run_pipeline(
            args.config,
            resume=True,
            analyze_only=True,
            model_ids_override=args.model_ids,
            port_override=args.port,
        )


if __name__ == "__main__":
    main()
