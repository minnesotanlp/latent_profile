arg="$1"

if [[ "$arg" -eq 1 ]]; then
    port=8001
    mid=8
elif [[ "$arg" -eq 2 ]]; then
    port=8002
    mid=19
elif [[ "$arg" -eq 3 ]]; then
    port=8003
    mid=20
else
    echo "Invalid argument: $arg"
    exit 1
fi

for bias in 0 1 2; do
  for topic in 0 3 6; do
    uv run python main.py --topic $topic --bias $bias --model-id $mid \
    --preference-respond 0 --qa-prompt 0 --save-dir last --port $port
    uv run python main.py --topic $topic --bias $bias --model-id $mid \
    --personality-respond 0 --qa-prompt 1 --save-dir last --port $port --one-five 1
  done
done


for bias1 in 0 1 2; do
  for bias2 in 0 1 2; do
    for topic in 0 3 6; do
      uv run python conversation.py --topic $topic --bias1 $bias1 --bias2 $bias2 --model-id $mid \
      --save-dir last --port $port --sample-size 1 &
    done
  done
done

wait