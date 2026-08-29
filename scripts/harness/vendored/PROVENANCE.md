# Vendored module provenance

Prompt text and stage design in this directory follow the DIALECTIC debate
agent (Bae et al., 2026), whose appendix documents its prompt set. Every
prompt is reproduced here in full and every deviation is listed below.

## prompts.py

Prompt text follows DIALECTIC's four prompt groups:

- argument generation
  (`ARGUMENT_GENERATION_SYSTEM_PROMPT`, `PRO_ARGUMENTS_USER_PROMPT`,
  `CONTRA_ARGUMENTS_USER_PROMPT`)
- argument critique (all four devil's-advocate prompts)
- argument evaluation
  (`CRITERIA_MAPPING`, `SINGLE_ARGUMENT_EVALUATION_SYSTEM_PROMPT`,
  `EVALUATE_SINGLE_ARGUMENT_USER_PROMPT`)
- argument refinement (all four refinement prompts)

Deliberate adaptations (documented for the paper):

1. **Pro/con length asymmetry fixed**: original pro prompt said "max. 100 words",
   contra said "2-3 sentences". Both now say "max. 100 words".
2. **Q&A wording -> evidence wording**: the harness has no question-tree /
   answering stage. Every occurrence of "questions and answers (about the
   company)" / "Q&A facts" is replaced by "evidence (about the company)".
   The replacement is applied uniformly to ALL prompts, so prompts remain
   IDENTICAL across experimental conditions (only the content of the
   `{evidence}` block varies).
3. **qa_indices dropped**: original prompts asked the model to return
   `qa_indices` referencing numbered Q&A pairs. Condition C0 feeds raw deck
   text (no indexable units), so to keep prompts identical across conditions
   the index-tracking instruction and output field were removed everywhere.
4. **JSON output instructions appended**: DIALECTIC used LangChain
   `with_structured_output`; this harness uses the plain `openai` client, so
   each prompt gets an explicit "Return a JSON object ..." suffix. The suffix
   is identical across conditions.
5. **Decision readout replaced**: DIALECTIC's decision stage compared mean
   pro vs. contra judge scores. In this study, the final decision is
   a forced-choice INVEST/PASS single-token readout with logprobs
   (new prompt, written for this harness; the score-comparison decision is
   still recorded as `score_decision` for reference).

## Flow logic (reimplemented) in ../debate.py

The stage flow reimplements DIALECTIC's pipeline:

- argument generation
- critique (incl. the "do not repeat the same critique" former-critique
  carry-over)
- evaluation (14-criterion sum scoring; top-K selection with even pro/contra
  split, odd slot to the side with the higher single best score; retry until
  exactly 14 scores)
- refinement
- decision (iteration loop, final scoring pass)
- argument-feedback formatting

Not reimplemented: web search and question-tree decomposition / answering,
which this study replaces with the verification stage and its evidence block.
The judge is `meta-llama/Llama-3.3-70B-Instruct-Turbo` (Together, temperature
0.0) in this study's configuration.

## Inherited prompt asymmetry

`CONTRA_ARGUMENTS_USER_PROMPT` carries the line "Lack of data is not a good
contra argument.", which the pro prompt has no counterpart to. It is inherited
unchanged and held identical across all five conditions, so it cannot account
for any between-condition difference, but it biases the contra side against
absence-of-evidence reasoning in every arm.
