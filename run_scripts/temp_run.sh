arg="$1"

if [[ "$arg" -eq 0 ]]; then
    port=8000
    model_id=
    for topic in 0 3 6; do
        uv run python conversation.py --topic $topic --bias1 0 --bias2 0 --model-id $model_id \
        --save-dir last --port $port --sample-size 1  --num-rounds 2 &
        uv run python conversation.py --topic $topic --bias1 0 --bias2 1 --model-id $model_id \
        --save-dir last --port $port --sample-size 1 --num-rounds 2 &
            uv run python conversation.py --topic $topic --bias1 0 --bias2 1 --model-id $model_id \
        --save-dir last --port $port --sample-size 1  --num-rounds 2 &
    done
elif [[ "$arg" -eq 1 ]]; then
    for topic in 0 3 6; do
        uv run python conversation.py --topic $topic --bias1 0 --bias2 0 --model-id $model_id \
        --save-dir last --port $port --sample-size 1  --num-rounds 2 &
        uv run python conversation.py --topic $topic --bias1 0 --bias2 1 --model-id $model_id \
        --save-dir last --port $port --sample-size 1 --num-rounds 2 &
            uv run python conversation.py --topic $topic --bias1 0 --bias2 1 --model-id $model_id \
        --save-dir last --port $port --sample-size 1  --num-rounds 2 &
    done
elif [[ "$arg" -eq 2 ]]; then
    for topic in 0 3 6; do
        uv run python conversation.py --topic $topic --bias1 0 --bias2 0 --model-id $model_id \
        --save-dir last --port $port --sample-size 1  --num-rounds 2 &
        uv run python conversation.py --topic $topic --bias1 0 --bias2 1 --model-id $model_id \
        --save-dir last --port $port --sample-size 1 --num-rounds 2 &
            uv run python conversation.py --topic $topic --bias1 0 --bias2 1 --model-id $model_id \
        --save-dir last --port $port --sample-size 1  --num-rounds 2 &
    done
elif [[ "$arg" -eq 3 ]]; then
    for topic in 0 3 6; do
        uv run python conversation.py --topic $topic --bias1 0 --bias2 0 --model-id $model_id \
        --save-dir last --port $port --sample-size 1  --num-rounds 2 &
        uv run python conversation.py --topic $topic --bias1 0 --bias2 1 --model-id $model_id \
        --save-dir last --port $port --sample-size 1 --num-rounds 2 &
            uv run python conversation.py --topic $topic --bias1 0 --bias2 1 --model-id $model_id \
        --save-dir last --port $port --sample-size 1  --num-rounds 2 &
    done
else
    echo "Invalid argument: $arg"
    exit 1
fi





wait