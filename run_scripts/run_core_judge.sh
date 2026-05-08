arg="$1"

if [[ "$arg" -eq 0 ]]; then
    port=8000
    mid=8
elif [[ "$arg" -eq 1 ]]; then
    port=8001
    mid=19
elif [[ "$arg" -eq 2 ]]; then
    port=8002
    mid=20
else
    echo "Invalid argument: $arg"
    exit 1
fi

for combinedbias in 0 1 2 3 4 5 6 7 8; do
  for topic in 0 3 6; do
    uv run python judge.py --topic $topic --judge-window 2 --combined-bias $combinedbias --model-id $mid --save-dir last --port $port
  done
done