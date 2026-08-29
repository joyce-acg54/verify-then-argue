#!/usr/bin/env python3
"""Emit the parallel agent-runner script that runs the aligner over all tasks.

Usage (from the repo root):
  python scripts/aligner/gen_workflow.py [--chunk 0 --chunk-size 30] \
      [--model haiku] [--only-missing] > /tmp/aligner_workflow.js

Reads data/aligner/tasks_index.json (written by prepare_tasks.py) and prints
a self-contained runner script: one structured-output Haiku 4.5 call per task,
with a per-task JSON schema that pins exact cardinalities (n arguments, one
critique-catch entry per canary) — the schema enforcement that the v1 smoke
test showed is necessary (free-form Haiku output produced malformed JSON and
per-premise argument entries).

The driver executes the printed script, writes each returned result to
data/aligner/raw/<task_id>.json, and runs collect.py --strict. Any runner that
sends PROMPT.md plus one task file and enforces the schema is equivalent. --only-missing skips tasks that already have a result
file (retry loop). --chunk/--chunk-size bound the per-workflow return size.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

TEMPLATE = """export const meta = {{
  name: 'aligner-run',
  description: 'Premise-tracing aligner: schema-enforced {model} agents, one per task',
  phases: [{{ title: 'Align', detail: '{n} tasks, chunk {chunk}' }}],
}}

const ROOT = {root}
const INDEX = {index}

function schemaFor(tid, m) {{
  const claimLink = {{
    type: "object", additionalProperties: false,
    properties: {{
      claim_idx: {{ type: "integer" }},
      relation: {{ type: "string", enum: ["asserts", "hedges", "contradicts"] }},
    }},
    required: ["claim_idx", "relation"],
  }}
  const canaryLink = {{
    type: "object", additionalProperties: false,
    properties: {{
      canary_id: m.canary_ids.length ? {{ type: "string", enum: m.canary_ids }} : {{ type: "string" }},
      relation: {{ type: "string", enum: ["asserts_falsified", "asserts_true", "hedged", "flagged"] }},
    }},
    required: ["canary_id", "relation"],
  }}
  return {{
    type: "object", additionalProperties: false,
    properties: {{
      task_id: {{ type: "string", enum: [tid] }},
      arguments: {{
        type: "array", minItems: m.n_args, maxItems: m.n_args,
        items: {{
          type: "object", additionalProperties: false,
          properties: {{
            idx: {{ type: "integer" }},
            premises: {{
              type: "array", minItems: 1,
              items: {{
                type: "object", additionalProperties: false,
                properties: {{
                  premise: {{ type: "string" }},
                  load_bearing: {{ type: "boolean" }},
                  claim_links: {{ type: "array", maxItems: m.n_claims === 0 ? 0 : 20, items: claimLink }},
                  canary_links: {{ type: "array", maxItems: m.canary_ids.length === 0 ? 0 : 10, items: canaryLink }},
                }},
                required: ["premise", "load_bearing", "claim_links", "canary_links"],
              }},
            }},
          }},
          required: ["idx", "premises"],
        }},
      }},
      canary_critique_catches: {{
        type: "array", minItems: m.canary_ids.length, maxItems: m.canary_ids.length,
        items: {{
          type: "object", additionalProperties: false,
          properties: {{
            canary_id: m.canary_ids.length ? {{ type: "string", enum: m.canary_ids }} : {{ type: "string" }},
            caught: {{ type: "boolean" }},
            evidence: {{ type: "string" }},
          }},
          required: ["canary_id", "caught", "evidence"],
        }},
      }},
    }},
    required: ["task_id", "arguments", "canary_critique_catches"],
  }}
}}

phase('Align')
const ids = Object.keys(INDEX)
const results = await parallel(ids.map(tid => () => {{
  const m = INDEX[tid]
  return agent(
    "You are the premise-tracing aligner for an NLP measurement study. " +
    "Read " + ROOT + "/scripts/aligner/PROMPT.md fully and follow it exactly. " +
    "Then Read the task file " + ROOT + "{tasks_rel}" + tid + ".json " +
    "(this is the {{TASK_JSON}}). The task has EXACTLY " + m.n_args + " arguments " +
    "(idx 0.." + (m.n_args - 1) + "); your output arguments array must have exactly " +
    "one entry per task argument, in idx order - premises belonging to an argument " +
    "go INSIDE that argument's premises list, never as separate argument entries. " +
    "It has " + m.n_claims + " claims and " + m.canary_ids.length + " canaries" +
    (m.canary_ids.length ? " (one canary_critique_catches entry per canary)" : " (canary_critique_catches must be [])") +
    ". Core rules: textual tracing only, never world knowledge; link claims only on " +
    "reused factual content, not topic overlap. Return the alignment via the " +
    "structured output tool.",
    {{ label: 'align:' + tid, model: {model_js}, schema: schemaFor(tid, m) }}
  ).then(r => ({{ tid: tid, result: r }}))
}}))

return results.filter(Boolean)"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=30)
    ap.add_argument("--model", default="haiku",
                    help="aligner model override (the paper's runs pin "
                         "Claude Haiku 4.5; see RUNNER.md)")
    ap.add_argument("--only-missing", action="store_true",
                    help="skip tasks that already have a result file")
    args = ap.parse_args()

    index_path = common.ALIGNER_DIR / "tasks_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    ids = sorted(index)
    if args.only_missing:
        ids = [t for t in ids if not common.result_path(t).is_file()]
    chunk = ids[args.chunk * args.chunk_size:(args.chunk + 1) * args.chunk_size]
    if not chunk:
        print("// no tasks in this chunk", file=sys.stderr)
        return 1
    sub = {t: index[t] for t in chunk}
    tasks_rel = "/" + str(common.TASKS_DIR.relative_to(common.REPO_ROOT)).replace("\\", "/") + "/"
    print(TEMPLATE.format(
        model=args.model,
        model_js=json.dumps(args.model),
        n=len(sub),
        chunk=args.chunk,
        root=json.dumps(str(common.REPO_ROOT)),
        index=json.dumps(sub, indent=1),
        tasks_rel=tasks_rel,
    ))
    n_total = len(ids)
    print(f"// chunk {args.chunk}: {len(sub)} of {n_total} pending tasks "
          f"(chunk-size {args.chunk_size})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
