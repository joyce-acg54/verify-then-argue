# Aligner execution model

The aligner executes Claude Haiku 4.5 against the versioned prompt
`aligner-v1` (`scripts/aligner/PROMPT.md`). The Python scripts here are the
deterministic shell around the model calls; the released code reads no
`ANTHROPIC_API_KEY`, and any driver that sends `PROMPT.md` plus one blinded
task file and enforces the prompt's output schema reproduces the step (a
plain Anthropic API loop works).

## Pipeline

```
results/<runs>.jsonl
  └─ prepare_tasks.py     -> data/aligner/tasks/<task_id>.json   (blinded)
       └─ [one Haiku 4.5 call per task]
            each call: PROMPT.md + task file -> structured JSON
            -> data/aligner/raw/<task_id>.json
  └─ collect.py           -> alignments.jsonl (unblinded join)
                          -> canary_propagation.csv  (E1 table)
                          -> claim_leakage.csv       (E2 table)
  └─ make_validation_sample.py -> REVIEW_alignments.csv (for the annotator)
                               -> validation_key.csv   (hidden)
```

## Agentic runner

`gen_workflow.py` emits the runner script used for the paper's runs (one
schema-checked agent call per task id), and `ingest_workflow_output.py`
writes the returned results into `data/aligner/raw/`. Any equivalent loop
reproduces the same artifacts:

1. List task ids from `data/aligner/tasks/`.
2. For each id, send `scripts/aligner/PROMPT.md` plus the task file to
   Haiku 4.5 and validate the response against the prompt's output schema.
3. Write the result to `data/aligner/raw/<id>.json`, then run
   `collect.py --strict`; RERUN-listed ids get one retry pass.

The pinned configuration for the paper's runs is Claude Haiku 4.5 with
prompt `aligner-v1`. The aligner is measurement-only and outside the
decision chain, so the model choice introduces no leakage confound. One of
216 transcripts (a 122-claim deck) was aligned on Claude Sonnet 4.5 after
Haiku twice emitted out-of-range claim indices, as disclosed in
`scripts/analysis/e2_analysis.py`.

## Human validation

1. After the first real batch: `make_validation_sample.py --n 180`.
2. The annotator fills the LABEL column in
   `data/aligner/REVIEW_alignments.csv` (y / hedge / n) without opening
   `validation_key.csv`.
3. Precision and the hard-negative miss rate are reported alongside the E1/E2
   endpoints as a bound on their interpretation. The endpoint confidence
   intervals are deck-level bootstrap intervals over observed statuses and do
   not themselves incorporate the aligner's error rate.
