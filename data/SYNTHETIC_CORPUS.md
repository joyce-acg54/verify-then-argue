# Synthetic corpus

This corpus is fully invented: the two companies ("Aquilon Robotics", SYN0001; "Meridian BioSensors", SYN0002), all numbers, and all claims are fictional and correspond to no real startup or fund data.

## Planted canaries

Each injected deck under `data/injected/<id>/deck_injected.txt` differs from its original by exactly one planted false fact, the analogue of a canary in the real corpus. Both are of the shape the real canaries take, an externally checkable assertion that would leave a public trace if it were true.

- **SYN0001** replaces the traction line `Aquilon has deployed 6 paid pilot systems with three logistics operators.` with `Aquilon was named Intralogistics Robot of the Year 2025 by the European Warehouse Automation Council.` The awarding body and the award are invented, so the claim is false and unfindable.
- **SYN0002** adds, after the field-study line, `The field study was conducted jointly with the European Institute for Occupational Heat Safety and published in 2025.` The institute and the publication are invented. The injection sits on the validation axis rather than the regulatory one so that it does not contradict the deck's own non-medical-device positioning, which would let a reader refute it without any retrieval.

## Claims files

`data/claims/<id>.json` is extracted from the **injected** deck, matching the invariant that `scripts/verify/run_all.py` enforces for canary accounts, so the planted claim enters the claim stream that C1, C2, C2-shuf and C3 read. The `verdict`, `source_reliability` and `consistency` values shipped here are illustrative placeholders in the correct schema rather than live verification output, since the invented companies have no web footprint to verify against. Running `scripts/verify/run_all.py` on this corpus overwrites these files with real verification output, which for invented companies will be `no_evidence` almost everywhere.

## Smoke test

```
python scripts/harness/run_batch.py \
    --targets data/targets_synthetic.csv \
    --conditions C0,C1,C2,C2shuf,C3 \
    --runs 1 --concurrency 4 \
    --c0-injected \
    --out results/synthetic_smoke.jsonl
```
