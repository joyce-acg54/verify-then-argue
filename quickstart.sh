#!/usr/bin/env bash
# Quickstart: run the five-condition debate experiment end to end on the
# included synthetic corpus. No confidential data is required.
#
#   ./quickstart.sh
#
# Requires OPENAI_API_KEY and TOGETHER_API_KEY in .env (copy .env.example).
# Expected cost is well under one US dollar for the two synthetic decks.

set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env ]; then
    echo "error: .env not found. Copy .env.example to .env and fill in" >&2
    echo "       OPENAI_API_KEY and TOGETHER_API_KEY." >&2
    exit 1
fi

if ! python3 -c "import openai, dotenv" 2>/dev/null; then
    echo "error: missing dependencies. Run: pip install -r requirements.txt" >&2
    exit 1
fi

OUT="results/synthetic_smoke.jsonl"
mkdir -p results

echo "Running 2 synthetic decks x 5 conditions x 1 run ..."
python3 scripts/harness/run_batch.py \
    --targets data/targets_synthetic.csv \
    --conditions C0,C1,C2,C2shuf,C3 \
    --runs 1 \
    --concurrency 4 \
    --c0-injected \
    --out "$OUT"

echo
echo "Wrote $OUT"
python3 - "$OUT" <<'PY'
import collections, json, sys

rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
by = collections.defaultdict(list)
for r in rows:
    by[r["condition"]].append(r["p_invest"])

print(f"{len(rows)} runs over {len({r['company_id'] for r in rows})} decks\n")
print("condition   mean P(invest)   n")
for cond in ("C0", "C1", "C2", "C2shuf", "C3"):
    if by[cond]:
        vals = by[cond]
        print(f"{cond:<11} {sum(vals)/len(vals):>13.3f}   {len(vals)}")
print(f"\ntotal cost: ${sum(r.get('cost_usd', 0) for r in rows):.4f}")
PY
