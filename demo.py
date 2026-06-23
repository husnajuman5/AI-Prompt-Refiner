"""
demo.py
-------
Runs the Iterative Prompt Refinement Algorithm end-to-end:

  1. Refines a single raw prompt and prints the full iteration log.
  2. Batch-validates the algorithm across the benchmark dataset.

Works out of the box with MockGenerativeModel (no API key / internet
required). To use a real model instead, set OPENAI_API_KEY or
ANTHROPIC_API_KEY and swap MockGenerativeModel() for OpenAIModel() /
AnthropicModel() below.
"""

import json
import os

from prompt_refiner import PromptRefiner, MockGenerativeModel, evaluate_output


def print_divider(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_single_example():
    print_divider("1) SINGLE PROMPT REFINEMENT")

    model = MockGenerativeModel()
    refiner = PromptRefiner(target_model=model, evaluate_fn=evaluate_output)

    raw_prompt = "Summarize this article."
    criteria = {
        "max_words": 120,
        "required_keywords": ["summary", "key", "point", "article"],
        "reference_text": "A concise neutral-tone summary covering the three key points of the article in under 120 words.",
    }

    result = refiner.refine(
        raw_prompt=raw_prompt,
        task_type="summarization",
        criteria=criteria,
        quality_threshold=0.6,
        max_iterations=5,
    )

    print(f"Raw prompt        : {raw_prompt!r}")
    print(f"Iterations used   : {len(result.iteration_log)}")
    print(f"Final quality score: {result.quality_score}")
    print("\n--- Iteration log ---")
    for entry in result.iteration_log:
        print(f"  Iter {entry['iteration']}: score={entry['score']:.3f}  "
              f"adjustments={entry['adjustments']}  reason={entry['failure_reason']}")

    print("\n--- Refined prompt ---")
    print(result.refined_prompt)
    print("\n--- Model output for refined prompt ---")
    print(result.final_output)


def run_dataset_validation():
    print_divider("2) BATCH VALIDATION ACROSS BENCHMARK DATASET")

    dataset_path = os.path.join(os.path.dirname(__file__), "dataset", "benchmark_prompts.json")
    with open(dataset_path) as f:
        dataset = json.load(f)

    model = MockGenerativeModel()
    refiner = PromptRefiner(target_model=model, evaluate_fn=evaluate_output)

    report = refiner.validate_on_dataset(dataset)

    print(f"{'Task Type':<18}{'Iterations':<12}{'Quality Score':<15}")
    print("-" * 45)
    for r in report["results"]:
        print(f"{r['task_type']:<18}{r['iterations_used']:<12}{r['quality_score']:<15}")

    print(f"\nAverage quality score across dataset: {report['average_quality_score']}")


if __name__ == "__main__":
    run_single_example()
    run_dataset_validation()
