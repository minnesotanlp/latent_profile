import subprocess
import signal
import time
import json
import requests
import argparse
import os
import psutil

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run-type", type=int, required=True, help="An integer to square")

args = parser.parse_args()
os.environ["HF_TOKEN"] = ""

if args.run_type == 0:
    print('0')
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,3"
    my_models = [3, 7]
    port = 8000
elif args.run_type == 1:
    print('1')
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    my_models = [4, 5, 6]
    port = 8001
elif args.run_type == 2:
    print('2')
    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    my_models = [8, 9]
    port = 8002
else:
    print('x' + 4)


def wait_for_vllm(base_url: str, timeout: int = 720, interval: int = 15):
    """Wait until the vLLM server is ready (within `timeout` seconds)."""
    url = base_url.rstrip("/") + "/v1/models"
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                print("✅ vLLM server is ready.")
                return True
        except requests.exceptions.RequestException:
            pass  # Server not up yet
        print("Waiting for vLLM to start...")
        time.sleep(interval)

    raise TimeoutError(f"❌ Timed out after {timeout} seconds waiting for vLLM.")

server_start_cmd = ["vllm", "serve", 
                    "MODEL_PLACEHOLDER", 
                    "--max-model-len", "2048", 
                    "--port", str(port)]

if args.run_type == 0:
    server_start_cmd.append("--pipeline-parallel-size")
    server_start_cmd.append("2")

main_args_cmd = ["uv", "run", "main.py", 
                 "--topic", "2", 
                 "--topic-correlation", "1", 
                 "--model-id", "NUM",
                 "--port", str(port)]

for model_idx in my_models:
    with open("variations/models.json", "r") as f:
        data = json.load(f)
    model_name = data[str(model_idx)]
    
    server_start_cmd[2] = model_name
    main_args_cmd[8] = str(model_idx)

    background_proc = subprocess.Popen(server_start_cmd, preexec_fn=os.setsid) #, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    wait_for_vllm("http://localhost:" + str(port))

    for topic in range(0, 9):
        for topic_correlation in range(0, 3):
            main_args_cmd[4] = str(topic)
            main_args_cmd[6] = str(topic_correlation)
            result = subprocess.run(main_args_cmd)

            # TODO Add checking here

    os.killpg(os.getpgid(background_proc.pid), signal.SIGTERM)

    try:
        background_proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        print("Background process didn't terminate in time. Forcibly killing.")
        background_proc.kill()