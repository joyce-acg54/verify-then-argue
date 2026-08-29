# Verify, Then Argue: Claim Verification in Investment Agent Debates

Code release for the EMNLP 2026 Industry Track paper by Joyce Ann Clarize Galang, Ahmed Mady, and Georg Groh ([OpenReview](https://openreview.net/forum?id=TRts7dV9tU)).

This repository contains the code for the verify-then-argue pipeline described in the paper. A verification stage extracts factual claims from a startup's pitch documents, routes each claim to a checking strategy, searches for external evidence, and produces a per-claim verdict label (together with source-reliability and internal-consistency signals). A fixed multi-agent debate harness then consumes that evidence and argues for or against investment under five experimental conditions: **C0** (raw document text, no verification), **C1** (extracted claims without verdicts), **C2** (claims with true verdicts), **C2-shuf** (claims with verdicts shuffled across claims, spelled `C2shuf` in the code), and **C3** (claims with verdicts plus source-reliability and consistency signals). Canary experiments inject planted falsehoods into the documents and probe whether they propagate into the debate's arguments and final decision.

## Directory map

```
scripts/
  verify/     Verification stage: claim extraction, per-claim routing,
              evidence search, and verdict assignment; also builds the
              per-company claims files consumed by the harness.
  canaries/   Planted-falsehood experiments: injection of canary claims
              into deck text and closed-book / recall probes.
  harness/    The debate itself (debate.py) plus the batch runner
              (run_batch.py) that iterates companies x conditions x runs.
  aligner/    Propagation measurement: aligns debate arguments back to
              source claims to measure which claims (and which canaries)
              made it into the arguments.
  analysis/   Statistics and figures for the paper (E1/E2 analyses,
              figure generation).
```

## Data layout contract

All stages read and write a shared `data/` tree keyed by a company identifier (the `account_id` column of the targets CSV):

- `data/documents/<account_id>/parsed/*.txt` — parsed document text. For condition C0 the batch runner picks the largest `.txt` in this directory (read-only, truncated to roughly 12k characters).
- `data/claims/<account_id>.json` — a non-empty JSON list of claim objects, each with keys `claim`, `verdict`, `source_reliability`, `consistency`, and `routing`. Conditions C1/C2/C2shuf/C3 read this file; C2shuf shuffles the `verdict` values across claims with a seed derived from `<account_id>:<seed>` so runs are reproducible.
- `data/injected/<account_id>/deck_injected.txt` — the canary-injected twin of the parsed deck, produced by `scripts/canaries/inject.py` and required when running C0 with `--c0-injected`.
- targets CSV — one row per company with an `account_id` column; passed to the batch runner via `--targets`.

Outputs go to `results/` (one JSONL line per company x condition x run) and intermediate results are cached under `data/cache/harness/`, which makes batch runs resumable.

## Running on the synthetic corpus

A synthetic corpus is included so that the debate harness and the verification stage run end to end without any confidential data; the canary probe, aligner, and analysis stages additionally require the confidential corpus artifacts described below and are included for inspection. See `data/SYNTHETIC_CORPUS.md` for what the two invented decks contain and which line each injected twin falsifies. Install the dependencies with `pip install -r requirements.txt`, then, with the environment keys below set, `./quickstart.sh` runs all five conditions on the two synthetic decks and prints a per-condition summary. It is a thin wrapper over:

```
python scripts/harness/run_batch.py \
    --targets data/targets_synthetic.csv \
    --conditions C0,C1,C2,C2shuf,C3 \
    --runs 1 --concurrency 4 \
    --c0-injected \
    --out results/synthetic_smoke.jsonl
```

`--c0-injected` runs C0 on the canary-injected twins that ship under `data/injected/` rather than the original decks, so the smoke run debates the same planted falsehoods the claims files carry (it hard-fails for any company without an injected deck). Regenerating those twins with `scripts/canaries/inject.py` needs the canary definition files, which are withheld, so the shipped twins are pre-built. Useful additional flags (see `run_batch.py --help`): `--T` (refinement iterations, default 2), `--K` (comma-separated top-K per iteration, default `5,4`), `--limit N` (cap the number of companies), and `--only ACCOUNT_ID,...` (restrict to specific companies).

## Running the verification stage

The verification stage runs on the same synthetic corpus:

```
python scripts/verify/run_all.py --accounts SYN0001,SYN0002 --parallel 2
```

This one needs `OPENAI_API_KEY` and `PERPLEXITY_API_KEY`, and it **overwrites** `data/claims/SYN0001.json` and `SYN0002.json`. Because the two companies are invented, live search finds nothing about them, so the rewritten files carry `no_evidence` almost everywhere and the debate conditions become degenerate. Copy the shipped claims files aside first if you want to run the harness afterwards.

## Released results

`results/` holds the per-run outputs behind the paper's experiments: `e1_grid.jsonl` (main E1 grid, seed 0), `e1_grid_seeds.jsonl` (three-seed replication, 40 decks x 5 conditions x seeds 0-2), `e1_grid_gpt4o.jsonl` (gpt-4o debater arm), `e2_grid.jsonl` (claim-leakage grid), and the derived CSVs. Note that `e1_propagation_by_condition.csv` and `e1_contrasts.csv` both describe the **canary-propagation** endpoint, not the decision endpoint; the decision contrasts reported in the paper are recomputed from the per-run grids by pairing on `company_hash`.

Because the decks are confidential, every free-text field was removed rather than scrubbed: `arguments_final[].text`, all of `critiques[]`, and `criterion_scores[].reasoning`. `company_id` is replaced by `company_hash`, a salted SHA-256 truncated to 16 hex characters, with a random salt that was never committed. Hashes are stable across files, so runs can be joined per deck. Every numeric and label field survives, so the P(invest) condition means and paired contrasts can be recomputed from these files. The propagation tables are derived by the aligner from argument text and therefore cannot be re-derived from them.

## What is not included and why

- The fund's pitch decks, the claims extracted from them, and all associated caches are confidential and cannot be released.
- The firm-side ingestion tooling (document collection and parsing against internal systems) is internal and not part of this release.
- A synthetic corpus is provided instead, following the data layout contract above, so that the debate harness and the verification stage run end to end.
- The `scripts/analysis/` stage reproduces the paper's statistics from full-run artifacts (aligner tables and canary metadata) that are not included; it is provided for methodological transparency. The canary probes and the aligner similarly require the confidential canary and run artifacts.
- A confidentiality sweep over paper sources and canary data runs before every paper build; its inputs are not part of this release.

## Environment keys

Copy `.env.example` to `.env` and fill in the values:

- `OPENAI_API_KEY` — debate harness (generation / judging / decision models).
- `TOGETHER_API_KEY` — debate harness (open-weights debater models).
- `PERPLEXITY_API_KEY` — live claim verification only (evidence search).
- `EXA_API_KEY` — live claim verification only (second search provider).

Only the first two are needed to run the debate harness on the synthetic corpus with pre-built claims files; the last two are needed only to re-run live verification.

## Citation

```bibtex
@inproceedings{galang2026verify,
  title     = {Verify, Then Argue: Claim Verification in Investment Agent Debates},
  author    = {Galang, Joyce Ann Clarize and Mady, Ahmed and Groh, Georg},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing: Industry Track},
  year      = {2026},
  address   = {Budapest, Hungary},
  publisher = {Association for Computational Linguistics},
  note      = {To appear}
}
```
