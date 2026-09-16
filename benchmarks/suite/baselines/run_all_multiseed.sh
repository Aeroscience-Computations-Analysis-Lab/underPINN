#!/bin/bash
# Runs the three eager-PyTorch-vs-JAX baseline scripts at 3 seeds each,
# 5000 epochs (matching Table 2's existing methodology). Waits for the
# concurrent multiseed_heavy_throughput_accuracy.py job to finish first so
# GPU contention does not skew wall-clock timings.
set -e
cd "$(dirname "$0")/../../.."

echo "Waiting for multiseed_heavy_throughput_accuracy.py to finish..."
while pgrep -f "multiseed_heavy_throughput_accuracy.py" > /dev/null; do
    sleep 15
done
echo "Heavy multiseed job done. Starting baseline comparisons."

for seed in 0 1 2; do
    echo "=== burgers seed=$seed ==="
    python3 benchmarks/suite/baselines/burgers_baselines.py \
        --epochs 5000 --seed $seed --skip torch_script torch_func_eager torch_func_compile
done

for seed in 0 1 2; do
    echo "=== pipe_flow seed=$seed ==="
    python3 benchmarks/suite/baselines/pipe_flow_baselines.py \
        --epochs 5000 --seed $seed
done

for seed in 0 1 2; do
    echo "=== ramp_ns seed=$seed ==="
    python3 benchmarks/suite/baselines/ramp_ns_baselines.py \
        --epochs 5000 --seed $seed
done

echo "ALL DONE"
