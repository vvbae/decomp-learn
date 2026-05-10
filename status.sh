#!/usr/bin/env bash
# Run from repo root to see what's running and what results exist.

REPO="$(cd "$(dirname "$0")" && pwd)"
OUTPUTS="$REPO/outputs/train"

echo "=========================================="
echo " TMUX SESSIONS"
echo "=========================================="
if tmux ls 2>/dev/null; then
    echo ""
    echo "Attach with: tmux attach -t <session-name>"
else
    echo "  (none running)"
fi

echo ""
echo "=========================================="
echo " EXPERIMENT RESULTS"
echo "=========================================="
printf "  %-35s %-12s %-10s %s\n" "experiment" "steps" "success%" "timestamp"
echo "  $(printf '%0.s-' {1..75})"

for exp_dir in "$OUTPUTS"/*/; do
    name=$(basename "$exp_dir")
    # find latest eval_results.json (highest step number)
    latest_json=$(find "$exp_dir/checkpoints" -name "eval_results.json" 2>/dev/null \
        | sort -t/ -k1,1 | tail -1)
    # find latest checkpoint step
    latest_ckpt=$(ls -d "$exp_dir/checkpoints"/[0-9]* 2>/dev/null | sort | tail -1)
    steps=$(basename "$latest_ckpt" 2>/dev/null || echo "?")

    if [[ -n "$latest_json" ]]; then
        rate=$(python3 -c "import json; d=json.load(open('$latest_json')); print(f\"{d['success_rate']*100:.1f}%\")" 2>/dev/null || echo "?")
        ts=$(python3 -c "import json; d=json.load(open('$latest_json')); print(d.get('timestamp','')[:16])" 2>/dev/null || echo "")
        printf "  %-35s %-12s %-10s %s\n" "$name" "$steps" "$rate" "$ts"
    else
        printf "  %-35s %-12s %-10s %s\n" "$name" "$steps" "(no eval)" ""
    fi
done

echo ""
echo "=========================================="
echo " SESSION NAMES (convention)"
echo "=========================================="
echo "  augment          data augmentation generation"
echo "  train-contact    contact policy (augmented data)"
echo "  train-e2e-50     E2E with 50 demos"
echo "  train-e2e-100    E2E with 100 demos"
echo "  train-e2e-200    E2E with 200 demos"
echo "  train-e2e-500    E2E with 500 demos"
echo "  train-cp-50      contact policy with 50 demos"
echo "  train-cp-100     contact policy with 100 demos"
echo "  train-cp-200     contact policy with 200 demos"
echo "  train-cp-500     contact policy with 500 demos"
