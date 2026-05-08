
model_id=0
# Outer loop: -1, 0, 1, 2
for corr in 0 1 2; do
  # Inner loop: 0 through 8
  for topic in {0..8}; do
    uv run python main.py --topic $topic --topic-correlation $corr --model-id $model_id \
    --personality-respond 0 --save-dir baseline --demo-prompt 1
    uv run python main.py --topic $topic --topic-correlation $corr --model-id $model_id \
    --personality-respond 0 --one-five 1 --qa-prompt 1 --save-dir baseline_one_five --one-five 1 --demo-prompt 1
    uv run python main.py --topic $topic --topic-correlation $corr --model-id $model_id \
    --personality-respond 0 --one-five 1 --qa-prompt 1 --save-dir one_five --one-five 1
  done
done