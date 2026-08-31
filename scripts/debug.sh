#!/usr/bin/env bash
# Terminal equivalents of .vscode/launch.json, for when you want pdb instead of VS Code.
#
#   ./scripts/debug.sh                 list the targets
#   ./scripts/debug.sh tour            all no-model stages, plain run
#   ./scripts/debug.sh pdb 5           drop into pdb at the top of tour stage 5
#   ./scripts/debug.sh model           tour stages 8-9 (loads the weights)
#   ./scripts/debug.sh infer           run_inference, 1 prompt, 64 tokens
#   ./scripts/debug.sh degenerate      the H(q)=0 control
#   ./scripts/debug.sh faith           run_faith generate, 1 question, 2 conditions
#   ./scripts/debug.sh report          run_faith report on an existing run (no model)
#   ./scripts/debug.sh tests           the whole test suite
#   ./scripts/debug.sh attach 5        wait for a VS Code attach, then run stage 5
#
# Everything runs from the repo root. No install and no PYTHONPATH needed -- each
# script does its own sys.path.insert(0, repo_root).

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="models/mlx-community--Qwen3-4B-Thinking-2507-8bit"
RUN="data/runs/20260820T071230Z_stage0cap16k"

case "${1:-help}" in
  tour)
    python3 scripts/debug_tour.py "${@:2}"
    ;;

  pdb)
    # -m pdb stops before the first line; `c` to run, or set breaks first.
    # Useful one-liners once you are at the (Pdb) prompt:
    #   b faithfulness/hints.py:165     break on build_messages
    #   b generation/entropy.py:55      break on the top_p/top_k/temp order
    #   b faithfulness/metrics.py:131   break on the bucket counting
    #   c                               continue    n  next    s  step in    p x  print
    python3 -m pdb scripts/debug_tour.py --stage "${2:-5}"
    ;;

  model)
    python3 scripts/debug_tour.py --with-model "${@:2}"
    ;;

  infer)
    python3 scripts/run_inference.py \
      --model "$MODEL" --prompts data/prompts/toy.jsonl \
      --limit 1 --max-tokens 64 --tag debug
    ;;

  degenerate)
    # H(q) must come out exactly 0.0000 while H(p_raw) stays well above it.
    python3 scripts/run_inference.py \
      --model "$MODEL" --limit 1 --max-tokens 32 \
      --temp 0.01 --top-k 1 --tag debug_degenerate
    ;;

  faith)
    python3 -u scripts/run_faith.py generate \
      --n 1 --conditions unhinted_plain,suggestion_False \
      --max-tokens 256 --tag debug
    ;;

  report)
    # No model, no key -- reads records.jsonl and writes tables.md + figures/.
    python3 scripts/run_faith.py report --run "$RUN"
    ;;

  tests)
    python3 -m pytest tests/ -q "${@:2}"
    ;;

  attach)
    # Blocks until VS Code's "Attach to a running process (port 5678)" connects.
    python3 -c "
import debugpy, runpy, sys
debugpy.listen(5678)
print('waiting for the VS Code debugger to attach on :5678 ...')
debugpy.wait_for_client()
sys.argv = ['debug_tour.py', '--stage', '${2:-5}']
runpy.run_path('scripts/debug_tour.py', run_name='__main__')
"
    ;;

  help|*)
    awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 && !/^#/ {exit}' "$0"
    echo
    echo "stages:"
    python3 scripts/debug_tour.py --list | sed 's/^/  /'
    ;;
esac
