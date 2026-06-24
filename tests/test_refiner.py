"""
tests/test_refiner.py
----------------------
Basic unit tests for the condition checks, augmentation functions, and the
end-to-end refinement loop. Run with: pytest tests/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prompt_refiner.refiner import (
    needs_role, needs_format_spec, needs_specificity, needs_reasoning, needs_examples,
    add_role, add_format_spec, add_specificity, add_reasoning, add_examples,
    PromptRefiner,
)
from prompt_refiner.models import MockGenerativeModel
from prompt_refiner.evaluation import evaluate_output, rule_based_score, semantic_score


# --- condition checks -------------------------------------------------------

def test_needs_role_true_for_bare_prompt():
    assert needs_role("Summarize this article.") is True


def test_needs_role_false_once_added():
    prompt = add_role("Summarize this article.", "summarization")
    assert needs_role(prompt) is False


def test_needs_format_spec_detects_existing_format():
    assert needs_format_spec("Return the result as JSON.") is False
    assert needs_format_spec("Summarize this article.") is True


def test_needs_specificity_flags_short_prompt():
    assert needs_specificity("Summarize this.") is True


def test_needs_specificity_false_after_added():
    prompt = add_specificity("Summarize this article in detail please now.")
    assert needs_specificity(prompt) is False


def test_needs_reasoning_only_for_reasoning_tasks():
    assert needs_reasoning("Write code to sort a list.", "code-generation") is True
    assert needs_reasoning("Write a poem.", "creative-writing") is False


def test_needs_examples_respects_injection_flag():
    examples = [{"in": "2+2", "out": "4"}]
    assert needs_examples("Solve this.", examples) is True
    prompt = add_examples("Solve this.", examples)
    assert needs_examples(prompt, examples) is False


# --- evaluation --------------------------------------------------------------

def test_rule_based_score_no_rules_returns_one():
    assert rule_based_score("anything", {}) == 1.0


def test_rule_based_score_keyword_match():
    score = rule_based_score("This is a summary of the key points.", {"required_keywords": ["summary"]})
    assert score == 1.0


def test_semantic_score_identical_text_is_high():
    score = semantic_score("the cat sat on the mat", "the cat sat on the mat")
    assert score > 0.9


def test_semantic_score_unrelated_text_is_low():
    score = semantic_score("quantum physics lecture notes", "chocolate chip cookie recipe")
    assert score < 0.3


# --- end-to-end refinement ----------------------------------------------------

def test_refine_improves_or_meets_threshold():
    model = MockGenerativeModel()
    refiner = PromptRefiner(target_model=model, evaluate_fn=evaluate_output)

    criteria = {
        "max_words": 120,
        "required_keywords": ["summary", "key", "point"],
        "reference_text": "A concise summary covering the key points of the article.",
    }

    result = refiner.refine(
        raw_prompt="Summarize this article.",
        task_type="summarization",
        criteria=criteria,
        quality_threshold=0.6,
        max_iterations=5,
    )

    assert len(result.iteration_log) >= 1
    # the refined prompt must contain more guidance than the raw prompt
    assert len(result.refined_prompt.split()) > len("Summarize this article.".split())
    # quality should never decrease iteration to iteration in the recorded best score
    assert result.quality_score >= result.iteration_log[0]["score"] - 1e-9


def test_validate_on_dataset_returns_average():
    model = MockGenerativeModel()
    refiner = PromptRefiner(target_model=model, evaluate_fn=evaluate_output)

    dataset = [
        {
            "task_type": "summarization",
            "raw_prompt": "Summarize this.",
            "criteria": {"required_keywords": ["summary"], "reference_text": "a summary"},
            "quality_threshold": 0.5,
            "max_iterations": 3,
        }
    ]
    report = refiner.validate_on_dataset(dataset)
    assert "average_quality_score" in report
    assert 0.0 <= report["average_quality_score"] <= 1.0
