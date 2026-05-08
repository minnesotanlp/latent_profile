arg="$1"

if [[ "$arg" -eq 0 ]]; then
    port=8000
    my_topics=(0 1 2)
elif [[ "$arg" -eq 1 ]]; then
    port=8001
    my_topics=(4 5)
elif [[ "$arg" -eq 2 ]]; then
    port=8002
    my_topics=(3 7)
elif [[ "$arg" -eq 3 ]]; then
    port=8003
    my_topics=(6 8)
else
    echo "Invalid argument: $arg"
    exit 1
fi

for bias1 in 0 1 2; do
  for bias2 in 0 1 2; do
    for topic in "${my_topics[@]}"; do
      uv run python conversation.py --topic $topic --bias1 $bias1 --bias2 $bias2 --model-id 2 \
      --save-dir last --port $port --sample-size 20 &
    done
  done
done

wait