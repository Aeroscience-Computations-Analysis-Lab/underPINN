#!/usr/bin/env bash
# Run the full reviewer-response benchmark suite.
#
#   bash benchmarks/suite/run_all.sh              # GPU, paper settings
#   bash benchmarks/suite/run_all.sh 2000         # shorter budget
#   bash benchmarks/suite/run_all.sh 5000 500     # 2nd arg: natural-gradient
#                                                     #   epoch budget (default 2000;
#                                                     #   Gauss-Newton is per-epoch
#                                                     #   expensive, so this is kept
#                                                     #   independent of $1)
#   ALLOW_CPU=1 bash benchmarks/suite/run_all.sh 50   # CPU smoke test only
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
echo " 1/7  Dispatch parity: why 12 ms/epoch vs 0.8 ms/epoch"
echo "=============================================================="
$PY benchmarks/suite/parity/dispatch_parity.py --epochs "$EPOCHS" $CPU_FLAG

echo
echo "=============================================================="
echo " 2/7  Strong baselines: eager / TorchScript / torch.func(+compile) / JAX"
echo "=============================================================="
$PY benchmarks/suite/baselines/burgers_baselines.py --epochs "$EPOCHS" $CPU_FLAG

echo
echo "=============================================================="
echo " 3/7  Feature ablation: gated attention, Fourier, FBPINN, RBA"
echo "=============================================================="
$PY benchmarks/suite/ablations/ablate_features.py --epochs "$EPOCHS" $CPU_FLAG

echo
echo "=============================================================="
echo " 4/7  Artificial-viscosity ablation: none / fixed / trainable"
echo "=============================================================="
$PY benchmarks/suite/ablations/ablate_artificial_viscosity.py \
    --epochs "$EPOCHS" $CPU_FLAG

echo
echo "=============================================================="
echo " 5/7  QR-DEIM-R ablation (Toro-3): none / rad / qr_deim resampling"
echo "=============================================================="
$PY benchmarks/suite/ablations/ablate_qr_deim.py --epochs "$EPOCHS" $CPU_FLAG

echo
echo "=============================================================="
echo " 6/7  QR-DEIM-R ablation (Ramp NS, the flagship RAR-D case)"
echo "=============================================================="
$PY benchmarks/suite/ablations/ablate_qr_deim_ramp_ns.py \
    --epochs "$EPOCHS" $CPU_FLAG

echo
echo "=============================================================="
echo " 7/7  Natural-gradient ablation: Adam vs Gauss-Newton"
echo "=============================================================="
NGD_EPOCHS="${2:-2000}"
$PY benchmarks/suite/ablations/ablate_natural_gradient.py \
    --epochs "$NGD_EPOCHS" $CPU_FLAG

echo
$PY benchmarks/suite/summarize.py
