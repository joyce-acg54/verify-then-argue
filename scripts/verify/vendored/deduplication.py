"""
Stage 2a: Deduplication — OpenAI text-embedding-3-small + cosine similarity.
Vendored from the earlier pipeline's deduplication.py (cost logging swapped in).
"""

import math
from collections import defaultdict

from openai import OpenAI

import costlog
from . import config
from .models import AtomicClaim
from .retry import with_retry

client = OpenAI(api_key=config.OPENAI_API_KEY)


def deduplicate_claims(claims: list[AtomicClaim]) -> list[AtomicClaim]:
    """Greedy embedding-similarity grouping; keeps the highest-confidence
    claim per group as canonical, marks the rest is_duplicate=True."""
    if len(claims) <= 1:
        return claims

    embeddings = get_embeddings([c.claim_text for c in claims])

    n       = len(claims)
    visited = set()
    groups: list[list[int]] = []

    for i in range(n):
        if i in visited:
            continue
        group = [i]
        visited.add(i)
        for j in range(i + 1, n):
            if j in visited:
                continue
            if cosine_similarity(embeddings[i], embeddings[j]) >= config.DEDUP_SIMILARITY_THRESHOLD:
                group.append(j)
                visited.add(j)
        groups.append(group)

    result = []
    for group in groups:
        if len(group) == 1:
            result.append(claims[group[0]])
            continue
        ranked = sorted([(claims[i], i) for i in group],
                        key=lambda x: x[0].support_confidence, reverse=True)
        canonical, _ = ranked[0]
        result.append(canonical)
        for dup, _ in ranked[1:]:
            dup.is_duplicate = True
            dup.canonical_id = canonical.claim_id
            result.append(dup)

    return result


def filter_unique(claims: list[AtomicClaim]) -> list[AtomicClaim]:
    return [c for c in claims if not c.is_duplicate]


def get_embeddings(texts: list[str]) -> list[list[float]]:
    try:
        response = with_retry(lambda: client.embeddings.create(
            model=config.EMBED_MODEL, input=texts))
        costlog.log_openai_response(response, stage="dedup_embed",
                                    model=config.EMBED_MODEL)
        return [item.embedding for item in response.data]
    except Exception:
        return [_ngram_vector(t) for t in texts]


def _ngram_vector(text: str, n: int = 3) -> list[float]:
    text   = text.lower()
    ngrams: dict[str, int] = defaultdict(int)
    for i in range(len(text) - n + 1):
        ngrams[text[i:i + n]] += 1
    norm = math.sqrt(sum(v * v for v in ngrams.values())) or 1.0
    vec  = [0.0] * 512
    for gram, count in ngrams.items():
        vec[hash(gram) % 512] += count / norm
    return vec


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot    = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1e-9
    norm_b = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (norm_a * norm_b)
