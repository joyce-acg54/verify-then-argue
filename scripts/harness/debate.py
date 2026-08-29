"""Lean multi-agent debate harness for "Verify, Then Argue" (EMNLP 2026 Industry).

Flow per company (adapted from DIALECTIC's stage design,
see vendored/PROVENANCE.md):
  generate 3 pro + 3 contra arguments
  -> T iterations of: devil's-advocate critique each -> judge scores each
     (14 criteria, 1-7 each, summed) -> select top K[t] (even pro/contra split)
     -> refine selected
  -> final judge scoring of surviving arguments
  -> forced-choice INVEST/PASS readout with logprobs -> P(invest)

Models (published config):
  generator + critic + refiner: gpt-4o-mini, temperature 0.5
  judge: meta-llama/Llama-3.3-70B-Instruct-Turbo via Together, temperature 0.0
  decision readout: gpt-4o-mini, logprobs=True, top_logprobs=20

No web search. No question-tree decomposition. The experimental condition
changes ONLY the evidence block (build_evidence_block); prompts are otherwise
identical across conditions.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
    UnprocessableEntityError,
)

HARNESS_DIR = Path(__file__).resolve().parent
REPO_ROOT = HARNESS_DIR.parents[1]
sys.path.insert(0, str(HARNESS_DIR))

import cost  # noqa: E402
from vendored import prompts as P  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

# Models are env-overridable so a stronger-debater robustness arm can be run
# without touching the published defaults (gpt-4o-mini debaters, Llama judge).
GEN_MODEL = os.environ.get("DEBATE_GEN_MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.environ.get("DEBATE_JUDGE_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo")
DECISION_MODEL = os.environ.get("DEBATE_DECISION_MODEL", "gpt-4o-mini")
GEN_TEMPERATURE = 0.5
JUDGE_TEMPERATURE = 0.0
RAW_TEXT_MAX_CHARS = 12_000
CONDITIONS = ("C0", "C1", "C2", "C2shuf", "C3")

_RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)

_clients_lock = threading.Lock()
_clients: dict[str, OpenAI] = {}


def _client(provider: str) -> OpenAI:
    with _clients_lock:
        if provider not in _clients:
            if provider == "openai":
                _clients[provider] = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            elif provider == "together":
                _clients[provider] = OpenAI(
                    api_key=os.environ["TOGETHER_API_KEY"],
                    base_url="https://api.together.xyz/v1",
                )
            else:
                raise ValueError(f"unknown provider {provider!r}")
        return _clients[provider]


def extract_json(text: str) -> dict:
    """Robustly pull a JSON object out of a model response."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


# ---------------------------------------------------------------------------
# Evidence block construction — the ONLY thing that differs across conditions
# ---------------------------------------------------------------------------

def build_evidence_block(
    condition: str,
    raw_text: str | None = None,
    claims: list[dict] | None = None,
    max_chars: int = RAW_TEXT_MAX_CHARS,
) -> str:
    """Build the evidence block for one experimental condition.

    C0      raw deck text (truncated to ~12k chars)
    C1      claims, text only
    C2      claims + verdict (belief/disbelief/ignorance/no_evidence)
    C2shuf  identical formatting to C2 — verdict shuffling is done by the CALLER
    C3      claims + verdict + source_reliability + consistency

    Claims are dicts: {"claim": str, "verdict": str|None,
                       "source_reliability": float|None, "consistency": float|None}
    """
    if condition == "C0":
        if raw_text is None:
            raise ValueError("C0 requires raw_text")
        return raw_text[:max_chars]

    if condition not in ("C1", "C2", "C2shuf", "C3"):
        raise ValueError(f"unknown condition {condition!r}")
    if not claims:
        raise ValueError(f"{condition} requires a non-empty claims list")

    lines = []
    for i, c in enumerate(claims):
        line = f"{i}: {c['claim']}"
        # Claims the pipeline never verified (routing unverifiable/inference/
        # normative) carry verdict None in the claims file; render them as the
        # label "unverified" — never the Python literal "None".
        if condition in ("C2", "C2shuf"):
            line += f" [verdict: {c.get('verdict') or 'unverified'}]"
        elif condition == "C3":
            rel = c.get("source_reliability")
            con = c.get("consistency")
            line += (
                f" [verdict: {c.get('verdict') or 'unverified'}"
                f" | source_reliability: {rel if rel is not None else 'n/a'}"
                f" | consistency: {con if con is not None else 'n/a'}]"
            )
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Debate harness
# ---------------------------------------------------------------------------

class DebateHarness:
    """One instance is reusable across companies/conditions (thread-safe)."""

    def __init__(
        self,
        T: int = 2,
        K: tuple[int, ...] | list[int] = (5, 4),
        n_pro: int = 3,
        n_contra: int = 3,
        stage_workers: int = 4,
        max_tries: int = 5,
    ):
        if len(K) != T:
            raise ValueError(f"len(K)={len(K)} must equal T={T}")
        self.T = T
        self.K = list(K)
        self.n_pro = n_pro
        self.n_contra = n_contra
        self.stage_workers = stage_workers
        self.max_tries = max_tries

    # -- low-level call ----------------------------------------------------

    def _chat(
        self,
        provider: str,
        model: str,
        system: str,
        user: str,
        temperature: float,
        seed: int | None,
        stage: str,
        meta: dict,
        json_mode: bool = True,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        max_tokens: int | None = None,
    ):
        """Chat call with retry/backoff + cost logging. Returns the response."""
        kwargs: dict = dict(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        if seed is not None:
            kwargs["seed"] = seed
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = top_logprobs or 20
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        last_err: Exception | None = None
        for attempt in range(self.max_tries):
            t0 = time.time()
            try:
                resp = _client(provider).chat.completions.create(**kwargs)
            except (BadRequestError, UnprocessableEntityError) as e:
                # Endpoint rejecting response_format — drop it and retry.
                # 400 (BadRequest) and 422 (Unprocessable) are siblings, not
                # parent/child: Together's JSON-mode grammar compiler raises
                # 422 "failed to compile grammar" on some judge inputs, which
                # must be handled like the 400 case. Validity is unaffected:
                # the same model/prompt/temperature runs without constrained
                # decoding; the extract_json + exactly-14-scores loop below is
                # the real robustness mechanism.
                if "response_format" in kwargs:
                    kwargs.pop("response_format")
                    last_err = e
                    continue
                raise
            except _RETRYABLE as e:
                last_err = e
                time.sleep(min(2**attempt + 0.5, 30))
                continue
            usage = resp.usage
            call_cost = cost.log_call(
                model=model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                stage=stage,
                company_id=meta.get("company_id", ""),
                condition=meta.get("condition", ""),
                seed=meta.get("seed"),
                latency_s=time.time() - t0,
            )
            with meta["lock"]:
                tk = meta["tokens"].setdefault(model, {"prompt": 0, "completion": 0})
                tk["prompt"] += usage.prompt_tokens
                tk["completion"] += usage.completion_tokens
                meta["cost_usd"] += call_cost
                meta["n_calls"] += 1
            return resp
        raise RuntimeError(f"LLM call failed after {self.max_tries} tries: {last_err}")

    def _map(self, fn, items):
        """Run fn over items in parallel (per-stage thread pool), order-preserving."""
        if not items:
            return []
        with ThreadPoolExecutor(max_workers=min(self.stage_workers, len(items))) as ex:
            return list(ex.map(fn, items))

    # -- stages -------------------------------------------------------------

    def _generate(self, side: str, evidence: str, seed: int | None, meta: dict) -> list[dict]:
        n = self.n_pro if side == "pro" else self.n_contra
        tmpl = P.PRO_ARGUMENTS_USER_PROMPT if side == "pro" else P.CONTRA_ARGUMENTS_USER_PROMPT
        user = tmpl.format(n_arguments=n, evidence=evidence)
        for attempt in range(self.max_tries):
            resp = self._chat(
                "openai", GEN_MODEL, P.ARGUMENT_GENERATION_SYSTEM_PROMPT, user,
                GEN_TEMPERATURE, seed, f"generate_{side}", meta,
            )
            try:
                args = extract_json(resp.choices[0].message.content)["arguments"]
                contents = [a["content"] for a in args if a.get("content")]
            except (json.JSONDecodeError, KeyError, TypeError):
                contents = []
            if len(contents) >= n:
                return [{"type": side, "content": c} for c in contents[:n]]
        raise RuntimeError(f"generation produced <{n} {side} arguments after retries")

    def _critique_one(self, arg: dict, evidence: str, seed: int | None, meta: dict) -> dict:
        if arg["type"] == "pro":
            system = P.DEVILS_ADVOCATE_PRO_SYSTEM_PROMPT
            tmpl = P.DEVILS_ADVOCATE_INDIVIDUAL_PRO_ARGUMENT_USER_PROMPT
        else:
            system = P.DEVILS_ADVOCATE_CONTRA_SYSTEM_PROMPT
            tmpl = P.DEVILS_ADVOCATE_INDIVIDUAL_CONTRA_ARGUMENT_USER_PROMPT
        user = tmpl.format(evidence=evidence, argument=arg["content"])
        if arg.get("former_critique"):
            # carried over from the predecessor pipeline's critique.py
            user += (
                "\nHere is your past critique - do not repeat the same critique "
                f"but find a new one:\n{arg['former_critique']}"
            )
        resp = self._chat(
            "openai", GEN_MODEL, system, user, GEN_TEMPERATURE, seed,
            f"critique_{arg['type']}", meta,
        )
        content = resp.choices[0].message.content
        try:
            arg["critique"] = str(extract_json(content)["critique"])
        except (json.JSONDecodeError, KeyError, TypeError):
            arg["critique"] = content.strip()  # fall back to raw text
        return arg

    def _score_one(self, arg: dict, seed: int | None, meta: dict) -> dict:
        """Judge: 14 criteria x 1-7, summed. Retries until exactly 14 scores."""
        critique = (
            "Critique of the argument: " + arg["critique"] if arg.get("critique") else ""
        )
        user = P.EVALUATE_SINGLE_ARGUMENT_USER_PROMPT.format(
            argument=arg["content"], critique=critique
        )
        for attempt in range(self.max_tries):
            resp = self._chat(
                "together", JUDGE_MODEL, P.SINGLE_ARGUMENT_EVALUATION_SYSTEM_PROMPT,
                user, JUDGE_TEMPERATURE, seed, "judge", meta,
            )
            try:
                scores = extract_json(resp.choices[0].message.content)["scores"]
            except (json.JSONDecodeError, KeyError, TypeError):
                scores = []
            if isinstance(scores, list) and len(scores) == len(P.CRITERIA_MAPPING):
                clean = []
                ok = True
                for i, s in enumerate(scores):
                    try:
                        val = max(1, min(7, int(s["score"])))
                    except (KeyError, TypeError, ValueError):
                        ok = False
                        break
                    clean.append(
                        {
                            "criterion": P.CRITERIA_MAPPING[i],
                            "score": val,
                            "reasoning": str(s.get("reasoning", "")),
                        }
                    )
                if ok:
                    arg["criterion_scores"] = clean
                    arg["score"] = sum(c["score"] for c in clean)
                    # format_argument_feedback (predecessor helpers.py)
                    arg["argument_feedback"] = "\n".join(
                        f"{c['criterion']}: {c['reasoning']} (Score: {c['score']})"
                        for c in clean
                    )
                    return arg
        raise RuntimeError("judge failed to return exactly 14 valid scores after retries")

    def _select_top_k(self, args: list[dict], k: int) -> list[dict]:
        """Top-K with even pro/contra split; odd slot to the side whose single
        best score is higher (vendored logic from the predecessor's evaluation.py)."""
        pros = sorted((a for a in args if a["type"] == "pro"), key=lambda a: a["score"], reverse=True)
        cons = sorted((a for a in args if a["type"] == "contra"), key=lambda a: a["score"], reverse=True)
        k_pro = k_con = k // 2
        if k % 2 == 1:
            pro_top = pros[0]["score"] if pros else -1
            con_top = cons[0]["score"] if cons else -1
            if pro_top >= con_top:
                k_pro += 1
            else:
                k_con += 1
        return pros[:k_pro] + cons[:k_con]

    def _refine_one(self, arg: dict, evidence: str, seed: int | None, meta: dict) -> dict:
        if arg["type"] == "pro":
            system = P.REFINE_PRO_ARGUMENT_SYSTEM_PROMPT
            tmpl = P.REFINE_PRO_ARGUMENTS_USER_PROMPT
        else:
            system = P.REFINE_CONTRA_ARGUMENT_SYSTEM_PROMPT
            tmpl = P.REFINE_CONTRA_ARGUMENTS_USER_PROMPT
        user = tmpl.format(
            evidence=evidence,
            argument=arg["content"],
            argument_feedback=arg.get("argument_feedback", ""),
        )
        resp = self._chat(
            "openai", GEN_MODEL, system, user, GEN_TEMPERATURE, seed,
            f"refine_{arg['type']}", meta,
        )
        content = resp.choices[0].message.content
        try:
            refined = str(extract_json(content)["refined_argument"])
        except (json.JSONDecodeError, KeyError, TypeError):
            refined = content.strip()  # fall back to raw text
        # next-iteration argument: refined content, critique becomes former_critique
        return {
            "type": arg["type"],
            "content": refined,
            "former_critique": arg.get("critique"),
            "prev_score": arg.get("score"),
        }

    def _decide(self, evidence: str, final_args: list[dict], seed: int | None, meta: dict) -> dict:
        """Forced-choice INVEST/PASS readout; P(invest) from first-token logprobs
        renormalized over the two options (prefix-matched, case-insensitive)."""

        def fmt(side: str) -> str:
            items = [a for a in final_args if a["type"] == side]
            return "\n".join(
                f"- (score {a.get('score', 'n/a')}/98) {a['content']}" for a in items
            ) or "(none)"

        user = P.DECISION_USER_PROMPT.format(
            evidence=evidence, pro_arguments=fmt("pro"), contra_arguments=fmt("contra")
        )
        resp = self._chat(
            "openai", DECISION_MODEL, P.DECISION_SYSTEM_PROMPT, user,
            0.0, seed, "decision", meta,
            json_mode=False, logprobs=True, top_logprobs=20, max_tokens=1,
        )
        token_lps = resp.choices[0].logprobs.content[0].top_logprobs
        p_invest_mass = p_pass_mass = 0.0
        seen = []
        for tlp in token_lps:
            tok = tlp.token.strip().upper()
            prob = math.exp(tlp.logprob)
            seen.append({"token": tlp.token, "prob": round(prob, 6)})
            if tok and "INVEST".startswith(tok):
                p_invest_mass += prob
            elif tok and "PASS".startswith(tok):
                p_pass_mass += prob
        total = p_invest_mass + p_pass_mass
        if total > 0:
            p_invest = p_invest_mass / total
        else:  # degenerate fallback: trust the sampled token
            sampled = resp.choices[0].message.content.strip().upper()
            p_invest = 1.0 if sampled.startswith("INV") else 0.0
        return {
            "decision": "invest" if p_invest > 0.5 else "pass",
            "p_invest": p_invest,
            "p_invest_mass": p_invest_mass,
            "p_pass_mass": p_pass_mass,
            "sampled_token": resp.choices[0].message.content,
            "top_logprobs": seen,
        }

    # -- full debate ---------------------------------------------------------

    def run(
        self,
        company_id: str,
        condition: str,
        evidence: str,
        seed: int = 0,
    ) -> dict:
        """Run one full debate; returns a JSON-serializable result dict."""
        meta = {
            "company_id": company_id,
            "condition": condition,
            "seed": seed,
            "tokens": {},
            "cost_usd": 0.0,
            "n_calls": 0,
            "lock": threading.Lock(),
        }
        timings: dict[str, float] = {}
        t_start = time.time()

        # Stage 1: generation (3 pro + 3 contra)
        t0 = time.time()
        results = self._map(
            lambda side: self._generate(side, evidence, seed, meta), ["pro", "contra"]
        )
        args = results[0] + results[1]
        timings["generate"] = time.time() - t0

        history = []
        # Stages 2-4 x T iterations: critique -> judge -> select top-K -> refine
        for it in range(self.T):
            t0 = time.time()
            args = self._map(lambda a: self._critique_one(a, evidence, seed, meta), args)
            timings[f"iter{it}_critique"] = time.time() - t0

            t0 = time.time()
            args = self._map(lambda a: self._score_one(a, seed, meta), args)
            timings[f"iter{it}_judge"] = time.time() - t0

            selected = self._select_top_k(args, self.K[it])

            t0 = time.time()
            refined = self._map(
                lambda a: self._refine_one(a, evidence, seed, meta), selected
            )
            timings[f"iter{it}_refine"] = time.time() - t0

            history.append(
                {
                    "iteration": it,
                    "k": self.K[it],
                    "scored_arguments": [
                        {k: v for k, v in a.items() if k != "argument_feedback"}
                        for a in args
                    ],
                    "selected_ids": [args.index(a) for a in selected],
                }
            )
            args = refined

        # Final scoring pass on the surviving (refined) arguments
        t0 = time.time()
        final_args = self._map(lambda a: self._score_one(a, seed, meta), args)
        timings["final_judge"] = time.time() - t0

        # Predecessor-style score comparison (kept for reference)
        pro_scores = [a["score"] for a in final_args if a["type"] == "pro"]
        con_scores = [a["score"] for a in final_args if a["type"] == "contra"]
        pro_avg = sum(pro_scores) / len(pro_scores) if pro_scores else 0.0
        con_avg = sum(con_scores) / len(con_scores) if con_scores else 0.0
        score_decision = "invest" if pro_avg > con_avg else "not_invest"

        # Forced-choice readout (the paper's decision variable)
        t0 = time.time()
        decision = self._decide(evidence, final_args, seed, meta)
        timings["decision"] = time.time() - t0
        timings["total"] = time.time() - t_start

        return {
            "company_id": company_id,
            "condition": condition,
            "seed": seed,
            "config": {
                "T": self.T,
                "K": self.K,
                "n_pro": self.n_pro,
                "n_contra": self.n_contra,
                "gen_model": GEN_MODEL,
                "judge_model": JUDGE_MODEL,
                "decision_model": DECISION_MODEL,
                "gen_temperature": GEN_TEMPERATURE,
                "judge_temperature": JUDGE_TEMPERATURE,
            },
            "evidence_chars": len(evidence),
            "arguments_final": [
                {
                    "type": a["type"],
                    "text": a["content"],
                    "score": a["score"],
                    "criterion_scores": a["criterion_scores"],
                }
                for a in final_args
            ],
            "critiques": [
                {
                    "iteration": h["iteration"],
                    "items": [
                        {"type": a["type"], "argument": a["content"], "critique": a.get("critique")}
                        for a in h["scored_arguments"]
                    ],
                }
                for h in history
            ],
            "history": history,
            "pro_avg_score": pro_avg,
            "contra_avg_score": con_avg,
            "score_decision": score_decision,
            "decision": decision["decision"],
            "p_invest": decision["p_invest"],
            "decision_detail": decision,
            "tokens": meta["tokens"],
            "cost_usd": round(meta["cost_usd"], 6),
            "n_llm_calls": meta["n_calls"],
            "timings": {k: round(v, 2) for k, v in timings.items()},
        }
