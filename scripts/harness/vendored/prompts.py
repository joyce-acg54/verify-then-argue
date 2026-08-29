"""Prompt templates reproducing the DIALECTIC debate agent
(Bae et al., 2026); see PROVENANCE.md.

Adaptations: pro/con length asymmetry fixed (both "max. 100 words"),
Q&A wording -> evidence wording, qa_indices removed, JSON output suffixes
added for the plain openai client. Prompts are IDENTICAL across experimental
conditions; only the content of the {evidence} block varies.
"""

# 14 evaluation criteria for argument quality
# (reproduced unchanged from DIALECTIC's argument-evaluation prompts)
CRITERIA_MAPPING = [
    "Local Acceptability",
    "Local Relevance",
    "Local Sufficiency",
    "Cogency",
    "Credibility",
    "Emotional Appeal",
    "Clarity",
    "Appropriateness",
    "Arrangement",
    "Effectiveness",
    "Global Acceptability",
    "Global Relevance",
    "Global Sufficiency",
    "Reasonableness",
]

_CRITERIA_BLOCK = """1. Local Acceptability - Are the premises believable and factually plausible given the provided evidence?
2. Local Relevance - Do the premises clearly contribute to supporting or rejecting the conclusion about investment?
3. Local Sufficiency - Do the premises provide enough support to justify the conclusion?
4. Cogency - Does the argument have premises that are acceptable, relevant, and sufficient to support the investment conclusion?
5. Credibility - Does the argument make the author appear credible and trustworthy to VC investors?
6. Emotional Appeal - Does the argument create emotions that make the VC investors more receptive?
7. Clarity - Does the argument use correct and widely unambiguous language as well as avoid deviation from the issue?
8. Appropriateness - Is the style of reasoning and language suitable for a professional VC investment discussion?
9. Arrangement - Is the argument well-structured, with a logical order of premises and conclusion?
10. Effectiveness - Does the argument succeed in persuading the VC investors toward or against investing?
11. Global Acceptability - Would most VCs consider it a valid/legitimate argument?
12. Global Relevance - Does the argument meaningfully contribute to resolving the overall investment question?
13. Global Sufficiency - Does the argument adequately anticipate and rebut the main counterarguments from the argument's stance?
14. Reasonableness - Does the argument resolve the issue in a way acceptable to the VC investors, balancing global acceptability, relevance, and sufficiency?"""

# ---------------------------------------------------------------------------
# Generation (DIALECTIC argument-generation prompts)
# ---------------------------------------------------------------------------

ARGUMENT_GENERATION_SYSTEM_PROMPT = """You are a very experienced investor at a top‑tier VC fund. You are also a great storyteller and can tell a compelling story.

"""

PRO_ARGUMENTS_USER_PROMPT = """Generate {n_arguments} pro arguments why this company is a good investment opportunity.

Each argument should be concise (max. 100 words) and backed by specific data from the evidence.

A good argument provides a unique perspective on the investment opportunity that addresses the following criteria:
{criteria}

Here is the evidence about the company:
{evidence}

Return a JSON object of the form {{"arguments": [{{"content": "<argument text>"}}, ...]}} with exactly {n_arguments} arguments.
"""

CONTRA_ARGUMENTS_USER_PROMPT = """Generate {n_arguments} contra arguments why this company is a bad investment opportunity.

Each argument should be concise (max. 100 words) and backed by specific data from the evidence.
Lack of data is not a good contra argument.

A good argument provides a unique perspective on the investment opportunity that addresses the following criteria:
{criteria}

Here is the evidence about the company:
{evidence}

Return a JSON object of the form {{"arguments": [{{"content": "<argument text>"}}, ...]}} with exactly {n_arguments} arguments.
"""

# ---------------------------------------------------------------------------
# Devil's advocate critique (DIALECTIC argument-critique prompts)
# ---------------------------------------------------------------------------

DEVILS_ADVOCATE_PRO_SYSTEM_PROMPT = """You are a very experienced VC investor against investing in the company. However, your colleague thinks it is a good investment opportunity.
Your job is to criticize the pro argument given by your colleague using the evidence about the company and defend your position.
Be direct to persuade your colleague not to invest in the company.
"""

DEVILS_ADVOCATE_INDIVIDUAL_PRO_ARGUMENT_USER_PROMPT = """Here is the evidence about the company:
{evidence}

Here is the argument you have to criticize to persuade the colleague not to invest in the company:
{argument}

Keep your critique concise in 3-4 sentences.

Return a JSON object of the form {{"critique": "<critique text>"}}.
"""

DEVILS_ADVOCATE_CONTRA_SYSTEM_PROMPT = """You are a very experienced VC investor in favor of investing in the company. However, your colleague thinks it is a bad investment opportunity.
Your job is to criticize the given contra argument given by your colleague using the evidence about the company and defend your position.
Be direct to persuade your colleague to invest in the company.
"""

DEVILS_ADVOCATE_INDIVIDUAL_CONTRA_ARGUMENT_USER_PROMPT = """Here is the evidence about the company:
{evidence}

Here is the argument you have to criticize to persuade the colleague to invest in the company:
{argument}

Keep your critique concise in 3-4 sentences.

Return a JSON object of the form {{"critique": "<critique text>"}}.
"""

# ---------------------------------------------------------------------------
# Evaluation / judge (DIALECTIC argument-evaluation prompts)
# ---------------------------------------------------------------------------

SINGLE_ARGUMENT_EVALUATION_SYSTEM_PROMPT = """You are an impartial LLM judge to evaluate the quality of an argument in the VC investment context. The goal of the argument is to support or reject a startup investment decision in a persuasive way.
The quality of an argument in the venture capital investment context should be evaluated along the following 14 dimensions. For each dimension, assign a score from 1 (Low) to 7 (High), and provide a short feedback (1 sentence) how to improve the score.

14 Dimensions:
{criteria}
""".format(criteria=_CRITERIA_BLOCK)

EVALUATE_SINGLE_ARGUMENT_USER_PROMPT = """Argument to evaluate:
{argument}
{critique}

Return ONLY a JSON object of the form {{"scores": [{{"criterion": "<dimension name>", "score": <integer 1-7>, "reasoning": "<one sentence>"}}, ...]}} with exactly 14 entries, one per dimension, in the order listed above.
"""

# ---------------------------------------------------------------------------
# Refinement (DIALECTIC argument-refinement prompts)
# ---------------------------------------------------------------------------

REFINE_PRO_ARGUMENT_SYSTEM_PROMPT = """You are a very experienced investor at a top‑tier VC fund. You are sure that the company is a good investment opportunity.
Your job is to revise your argument to reach better argument quality scores.
"""

REFINE_CONTRA_ARGUMENT_SYSTEM_PROMPT = """You are a very experienced investor at a top‑tier VC fund. You are sure that the company is a bad investment opportunity.
Your job is to revise your argument to reach better argument quality scores.
"""

REFINE_PRO_ARGUMENTS_USER_PROMPT = """Here is the evidence about the company:
{evidence}

Here is your previous argument:
{argument}

Here are the argument quality scores (1-7) to your previous argument:
{argument_feedback}

Refine your argument by improving argument quality scores.

Return a JSON object of the form:
{{
    "refined_argument": "Refined Argument (one paragraph in max. 2 sentences)"
}}
"""

REFINE_CONTRA_ARGUMENTS_USER_PROMPT = """Here is the evidence about the company:
{evidence}

Here are the argument quality scores (1-7) to your previous argument:
{argument_feedback}

Here is your previous argument:
{argument}

Refine your argument by improving argument quality scores.

Return a JSON object of the form:
{{
    "refined_argument": "Refined Argument (one paragraph in max. 2 sentences)"
}}
"""

# ---------------------------------------------------------------------------
# Forced-choice decision readout (NEW for this harness; replaces DIALECTIC's
# score-comparison decision in DIALECTIC's decision stage)
# ---------------------------------------------------------------------------

DECISION_SYSTEM_PROMPT = """You are a very experienced investor at a top‑tier VC fund making a final, forced-choice investment decision."""

DECISION_USER_PROMPT = """Here is the evidence about the company:
{evidence}

Here are the final PRO arguments (with argument quality scores out of 98):
{pro_arguments}

Here are the final CONTRA arguments (with argument quality scores out of 98):
{contra_arguments}

Based on the evidence and the arguments above, decide whether the fund should invest in this company. You must choose. Respond with exactly one word: INVEST or PASS.

Decision:"""

# Fill the shared criteria block into the generation prompts once, so callers
# only supply {n_arguments} and {evidence}.
PRO_ARGUMENTS_USER_PROMPT = PRO_ARGUMENTS_USER_PROMPT.replace("{criteria}", _CRITERIA_BLOCK)
CONTRA_ARGUMENTS_USER_PROMPT = CONTRA_ARGUMENTS_USER_PROMPT.replace("{criteria}", _CRITERIA_BLOCK)
