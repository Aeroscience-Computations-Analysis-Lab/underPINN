#!/usr/bin/env bash
# Run the full reviewer-response benchmark suite.
#
#   bash benchmarks/rebuttal/run_all.sh              # GPU, paper settings
#   bash benchmarks/rebuttal/run_all.sh 2000         # shorter budget
#   ALLOW_CPU=1 bash benchmarks/rebuttal/run_all.sh 50   # CPU smoke test only
#
# Every script refuses to run on CPU unless ALLOW_CPU=1, so a CPU fallback can
# never be mistaken for a GPU measurement.
set -euo pipefail

cd "$(dirname "$0")/../.."
EPOCHS="${1:-5000}"
PY="${PYTHON:-python3}"
CPU_FLAG=""
if [[ "${ALLOW_CPU:-0}" == "1" ]]; then
  CPU_FLAG="--allow-cpu"
  echo "!! ALLOW_CPU=1 -- results are a correctness smoke test, NOT paper numbers"
fi

echo "=============================================================="
echo " 1/4  Dispatch parity: why 12 ms/epoch vs 0.8 ms/epoch"
echo "=============================================================="
$PY benchmarks/rebuttal/parity/dispatch_parity.py --epochs "$EPOCHS" $CPU_FLAG

echo
echo "=============================================================="
echo " 2/4  Strong baselines: eager / TorchScript / torch.func(+compile) / JAX"
echo "=============================================================="
$PY benchmarks/rebuttal/baselines/burgers_baselines.py --epochs "$EPOCHS" $CPU_FLAG

echo
echo "=============================================================="
echo " 3/4  Feature ablation: gated attention, Fourier, FBPINN, RBA"
echo "=============================================================="
$PY benchmarks/rebuttal/ablations/ablate_features.py --epochs "$EPOCHS" $CPU_FLAG

echo
echo "=============================================================="
echo " 4/4  Artificial-viscosity ablation: none / fixed / trainable"
echo "=============================================================="
$PY benchmarks/rebuttal/ablations/ablate_artificial_viscosity.py \
    --epochs "$EPOCHS" $CPU_FLAG

echo
$PY benchmarks/rebuttal/summarize.py
