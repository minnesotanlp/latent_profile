arg="$1"

if [[ "$arg" -eq 0 ]]; then
    port=8000
    my_topics=(0 1 2)
elif [[ "$arg" -eq 1 ]]; then
    port=8001
    my_topics=(3 4)
elif [[ "$arg" -eq 2 ]]; then
    port=8002
    my_topics=(5 6)
elif [[ "$arg" -eq 3 ]]; then
    port=8003
    my_topics=(7 8)
else
    echo "Invalid argument: $arg"
    exit 1
fi

for bias in 0 1 2; do
  for topic in "${my_topics[@]}"; do
    uv run python main.py --topic $topic --bias $bias --model-id 2 \
    --preference-respond 0 --qa-prompt 0 --save-dir last --port $port
  done
done