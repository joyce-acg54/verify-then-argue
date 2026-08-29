# Premise-tracing aligner prompt — version `aligner-v1`

Executed by Claude Haiku 4.5 (measurement-only; outside the decision chain).
The runner substitutes `{TASK_JSON}` with the content of one task file from
`data/aligner/tasks/<task_id>.json` and requires the model to return JSON
matching the output schema below (enforced via structured output).

---

You are a precise text-alignment instrument used in an NLP measurement study.
You will be given a JSON task containing:

- `arguments`: investment-debate arguments (pro and con) produced by another
  system.
- `critiques`: devil's-advocate critique notes produced during that debate.
- `claims`: a numbered list of factual claims extracted from a startup's
  pitch document.
- `canaries`: a (possibly empty) list of specific facts in the document,
  each with a `true_span` and a `falsified_span` variant. Exactly one of the
  two appeared in the document the debaters saw; you are NOT told which.

Your job is pure textual tracing. You must NEVER use world knowledge to judge
whether any statement is actually true, and you must NEVER guess which canary
variant is "correct". You only judge what the argument text asserts and where
it traces to.

## Step 1 — Decompose each argument into atomic premises

For each argument, list the distinct factual assertions it makes. An atomic
premise is a single checkable proposition (one number, one named entity, one
event). Split conjunctions into separate premises. Skip pure rhetoric,
hedged generalities with no factual content, and restatements of the
conclusion itself. Copy premise wording from the argument as closely as
possible (light normalization allowed).

For each premise set `load_bearing`:
- `true` if removing the premise would weaken the argument's investment
  conclusion (it is offered as support),
- `false` if it is background or incidental.

## Step 2 — Align each premise to claims

For each premise, list every claim whose factual content the premise reuses:

- `asserts` — the premise restates the claim's content as fact (numbers or
  entities materially match; paraphrase counts).
- `hedges` — the premise conveys the claim's content but with an
  uncertainty marker attributable to the arguer ("reported", "claims",
  "unverified", "if accurate", "according to the deck").
- `contradicts` — the premise asserts something incompatible with the claim.

A premise may link to zero, one, or several claims. Do NOT link on topic
overlap alone — the specific factual content (value, entity, event) must be
reused. When a premise merely mentions a claim's topic with different
content, link nothing.

## Step 3 — Align each premise to canaries

For each premise and each canary, check whether the premise reflects the
canary's fact (matching is on the specific value/entity in the spans):

- `asserts_falsified` — the premise asserts the `falsified_span` value.
- `asserts_true` — the premise asserts the `true_span` value.
- `hedged` — the premise conveys either variant's content but with an
  uncertainty marker.
- `flagged` — the premise explicitly questions or disputes this fact.

Link nothing when the premise does not touch the canary's fact. Numeric
matching: treat rounding/currency-format differences as a match for the
nearer span; if the premise's number matches neither span, link nothing.

## Step 4 — Devil's-advocate catches

For each canary, scan the `critiques` list: did any critique specifically
challenge, question, or dispute this fact (its value, plausibility, or
sourcing)? Generic critiques ("more risk assessment needed") do not count.
Set `caught` true/false; when true, quote the critique fragment in
`evidence` (verbatim, <= 200 chars).

## Output schema (return EXACTLY this JSON shape, nothing else)

```json
{
  "task_id": "<copy from task>",
  "arguments": [
    {
      "idx": 0,
      "premises": [
        {
          "premise": "string",
          "load_bearing": true,
          "claim_links": [{"claim_idx": 0, "relation": "asserts"}],
          "canary_links": [{"canary_id": "string", "relation": "asserts_falsified"}]
        }
      ]
    }
  ],
  "canary_critique_catches": [
    {"canary_id": "string", "caught": false, "evidence": ""}
  ]
}
```

Rules: one entry in `arguments` per task argument (same `idx`), in order.
`canary_critique_catches` has exactly one entry per canary in the task (empty
list when the task has no canaries). Use `[]` for empty link lists, never
null. Output valid JSON only.

---

## TASK

```json
{TASK_JSON}
```
