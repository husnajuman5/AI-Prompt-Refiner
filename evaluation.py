"""
evaluation.py
-------------
Quality/accuracy scoring for generated outputs.

Two complementary signals are combined into a single 0.0-1.0 score:

1. rule_based_score   - regex/keyword checks against explicit criteria
                          (format compliance, required keywords, length).
2. semantic_score      - TF-IDF + cosine similarity between the generated
                          output and a reference description of what a
                          good answer should contain. This needs no
                          internet access / model download, unlike a full
                          sentence-embedding model, which keeps the whole
                          pipeline runnable offline.
"""

from __future__ import annotations
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def rule_based_score(output: str, criteria: dict) -> float:
    """
    criteria supports the following optional keys:
      - max_words (int)
      - min_words (int)
      - required_keywords (list[str])   -- at least one must match (case-insensitive)
      - required_format (str)           -- "json" | "bullets" | None
    Returns a score in [0.0, 1.0].
    """
    checks = []

    words = output.split()
    if "max_words" in criteria:
        checks.append(len(words) <= criteria["max_words"])
    if "min_words" in criteria:
        checks.append(len(words) >= criteria["min_words"])

    if criteria.get("required_keywords"):
        text_lower = output.lower()
        checks.append(any(kw.lower() in text_lower for kw in criteria["required_keywords"]))

    fmt = criteria.get("required_format")
    if fmt == "json":
        checks.append(bool(re.search(r"[{[].*[}\]]", output, re.S)))
    elif fmt == "bullets":
        checks.append(bool(re.search(r"^[\s]*[-*]\s+", output, re.M)))

    if not checks:
        return 1.0  # no rules supplied -> don't penalize
    return sum(checks) / len(checks)


def semantic_score(output: str, reference_text: str) -> float:
    """
    TF-IDF cosine similarity between the generated output and a reference
    description of the ideal answer. Offline-safe (no pretrained model
    download required).
    """
    if not output.strip() or not reference_text.strip():
        return 0.0
    vectorizer = TfidfVectorizer().fit([output, reference_text])
    vectors = vectorizer.transform([output, reference_text])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return float(similarity)


def evaluate_output(output: str, criteria: dict, rule_weight: float = 0.4, semantic_weight: float = 0.6) -> float:
    """
    Combined quality score used by the refinement loop.
    criteria must contain "reference_text" for the semantic component, plus
    any of the rule_based_score keys above.
    """
    r_score = rule_based_score(output, criteria)
    s_score = semantic_score(output, criteria.get("reference_text", ""))
    return rule_weight * r_score + semantic_weight * s_score
