"""
refiner.py
----------
Core implementation of the Iterative Prompt Refinement Algorithm (IPRA).

Inputs
------
raw_prompt          : str   - user's initial, unrefined prompt
task_type           : str   - e.g. "summarization", "code-generation", "qa", "creative-writing"
target_model        : object with .generate(prompt) -> str
few_shot_examples   : list[dict] | None - [{"in": ..., "out": ...}, ...]
quality_threshold   : float - minimum acceptable score in [0, 1]
max_iterations      : int   - cap on refinement loop iterations
criteria            : dict  - passed through to evaluate_output() (see evaluation.py)

Outputs
-------
RefinementResult dataclass containing:
    refined_prompt, final_output, quality_score, iteration_log
"""

from __future__ import annotations
from dataclasses import dataclass, field
import re
import time


@dataclass
class RefinementResult:
    refined_prompt: str
    final_output: str
    quality_score: float
    iteration_log: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Condition checks (each gates whether a prompt-engineering element is added)
# ---------------------------------------------------------------------------

def needs_role(prompt: str) -> bool:
    return not re.search(r"you are|as an? expert", prompt, re.I)


def needs_reasoning(prompt: str, task_type: str) -> bool:
    reasoning_tasks = {"qa", "question-answering", "code-generation", "math"}
    return task_type.lower() in reasoning_tasks and not re.search(r"step by step|think through", prompt, re.I)


def needs_examples(prompt: str, few_shot_examples) -> bool:
    return bool(few_shot_examples) and "Input:" not in prompt


def needs_format_spec(prompt: str) -> bool:
    return not re.search(r"json|bullet|word limit|format", prompt, re.I)


def needs_specificity(prompt: str, min_words: int = 8) -> bool:
    has_constraint_word = re.search(r"audience|tone|words|length", prompt, re.I)
    return len(prompt.split()) < min_words or not has_constraint_word


def missing_required_keywords(output: str, criteria: dict) -> list:
    """Which of criteria['required_keywords'] are absent from the output so far."""
    required = criteria.get("required_keywords") or []
    output_lower = output.lower()
    return [kw for kw in required if kw.lower() not in output_lower]


# ---------------------------------------------------------------------------
# Augmentation functions (each adds exactly one prompt-engineering element)
# ---------------------------------------------------------------------------

def add_role(prompt: str, task_type: str) -> str:
    return f"You are an expert {task_type} assistant.\n{prompt}"


def add_reasoning(prompt: str) -> str:
    return f"{prompt}\nThink through this step by step before giving your final answer."


def add_examples(prompt: str, examples) -> str:
    block = "\n".join(f"Input: {e['in']}\nOutput: {e['out']}" for e in examples)
    return f"{block}\n\n{prompt}"


def add_format_spec(prompt: str) -> str:
    return f"{prompt}\nRespond in a clear, well-structured format (use bullet points where useful)."


def add_specificity(prompt: str) -> str:
    return f"{prompt}\nBe specific: state the intended audience, desired tone, and a target length."


def add_keyword_emphasis(prompt: str, missing_keywords: list) -> str:
    return f"{prompt}\nBe sure to explicitly mention: {', '.join(missing_keywords)}."


# ---------------------------------------------------------------------------
# Failure diagnosis (used to decide what to try next when a candidate scores low)
# ---------------------------------------------------------------------------

def diagnose_failure(output: str, score: float) -> str:
    if score < 0.3:
        return "off_topic_or_empty"
    if score < 0.6:
        return "missing_structure_or_detail"
    return "near_threshold"


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

class PromptRefiner:
    def __init__(self, target_model, evaluate_fn, rate_limit_seconds: float = 0.0):
        """
        target_model : object exposing .generate(prompt: str) -> str
        evaluate_fn  : callable(output: str, criteria: dict) -> float in [0, 1]
        """
        self.target_model = target_model
        self.evaluate_fn = evaluate_fn
        self.rate_limit_seconds = rate_limit_seconds

    def refine(
        self,
        raw_prompt: str,
        task_type: str,
        criteria: dict,
        few_shot_examples=None,
        quality_threshold: float = 0.8,
        max_iterations: int = 5,
    ) -> RefinementResult:
        # Step 1: initialize
        prompt = raw_prompt.strip()
        iteration = 0
        best_score = 0.0
        best_prompt = prompt
        best_output = ""
        iteration_log = []
        examples_injected = False

        # Step 3: main refinement loop
        previous_score = None
        while iteration < max_iterations and best_score < quality_threshold:
            adjustments = []

            if needs_role(prompt):
                prompt = add_role(prompt, task_type)
                adjustments.append("added_role")

            if needs_reasoning(prompt, task_type):
                prompt = add_reasoning(prompt)
                adjustments.append("added_reasoning")

            if not examples_injected and needs_examples(prompt, few_shot_examples):
                prompt = add_examples(prompt, few_shot_examples)
                examples_injected = True
                adjustments.append("added_examples")

            if needs_format_spec(prompt):
                prompt = add_format_spec(prompt)
                adjustments.append("added_format_spec")

            if needs_specificity(prompt):
                prompt = add_specificity(prompt)
                adjustments.append("added_specificity")

            # If the base augmentations are already all present (no adjustments
            # made this pass) but we're still below threshold, target the
            # specific missing required_keywords from the last evaluated output.
            if not adjustments and best_output:
                missing = missing_required_keywords(best_output, criteria)
                if missing:
                    prompt = add_keyword_emphasis(prompt, missing)
                    adjustments.append("added_keyword_emphasis")

            # Call target model
            output = self.target_model.generate(prompt)

            # Evaluate
            score = self.evaluate_fn(output, criteria)

            if score > best_score:
                best_score, best_prompt, best_output = score, prompt, output

            failure_reason = diagnose_failure(output, score) if score < quality_threshold else "passed"

            iteration_log.append({
                "iteration": iteration,
                "prompt": prompt,
                "output": output,
                "score": round(score, 4),
                "adjustments": adjustments,
                "failure_reason": failure_reason,
            })

            if best_score >= quality_threshold:
                break

            # Stagnation guard: no new adjustments were available AND no score
            # improvement this pass means further identical iterations would
            # be wasted model calls, so stop early.
            if not adjustments and previous_score is not None and score <= previous_score:
                break

            previous_score = score
            iteration += 1
            if self.rate_limit_seconds:
                time.sleep(self.rate_limit_seconds)

        return RefinementResult(
            refined_prompt=best_prompt,
            final_output=best_output,
            quality_score=round(best_score, 4),
            iteration_log=iteration_log,
        )

    def validate_on_dataset(self, dataset: list[dict]) -> dict:
        """
        FOR loop: batch-runs refine() across every benchmark sample in
        `dataset` (each item must provide raw_prompt, task_type, criteria,
        quality_threshold, max_iterations, few_shot_examples).
        Returns per-sample results plus the average quality score.
        """
        results = []
        for sample in dataset:
            result = self.refine(
                raw_prompt=sample["raw_prompt"],
                task_type=sample["task_type"],
                criteria=sample["criteria"],
                few_shot_examples=sample.get("few_shot_examples"),
                quality_threshold=sample.get("quality_threshold", 0.8),
                max_iterations=sample.get("max_iterations", 5),
            )
            results.append({
                "task_type": sample["task_type"],
                "raw_prompt": sample["raw_prompt"],
                "quality_score": result.quality_score,
                "iterations_used": len(result.iteration_log),
                "refined_prompt": result.refined_prompt,
                "final_output": result.final_output,
            })

        average_score = sum(r["quality_score"] for r in results) / len(results) if results else 0.0
        return {"results": results, "average_quality_score": round(average_score, 4)}
