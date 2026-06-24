# Prompt Engineering for Enhancing Generative AI Output Quality and Accuracy

Source code implementing the **Iterative Prompt Refinement Algorithm (IPRA)**:
it takes a vague, under-specified draft prompt and automatically rewrites it
(adding role, reasoning instructions, few-shot examples, format constraints,
and specificity) in a loop, scoring each candidate output, until a target
quality threshold is reached or a maximum number of attempts is used up.

## Features

- Iterative prompt refinement
- Quality evaluation
- Dataset benchmarking
- Mock and real model support

## Project Structure

```
prompt_refiner/
├── prompt_refiner/
│   ├── __init__.py        # package exports
│   ├── refiner.py         # core algorithm: condition checks, augmentations, main loop
│   ├── models.py          # MockGenerativeModel (offline) + OpenAIModel / AnthropicModel adapters
│   └── evaluation.py      # rule-based + TF-IDF semantic scoring
├── dataset/
│   └── benchmark_prompts.json   # 5 benchmark draft prompts used to validate the algorithm
├── tests/
│   └── test_refiner.py    # unit tests (pytest)
├── demo.py                # runnable end-to-end demo
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

No API key is required to run the demo — it uses `MockGenerativeModel`,
a deterministic offline stand-in for a real LLM that scales its output
quality with how many prompt-engineering elements (role, examples, format,
reasoning, constraints) are present in the prompt. This keeps the whole
algorithm runnable and testable without internet access or API costs.

## Running the demo

```bash
python demo.py
```

This will:
1. Refine a single raw prompt ("Summarize this article.") and print the
   full iteration log (score, adjustments made, and the final refined
   prompt + output).
2. Batch-validate the algorithm across `dataset/benchmark_prompts.json`
   and print a per-task quality score table plus the dataset average.

## Running the tests

```bash
pytest tests/
```

## Using a real generative AI model instead of the mock

```python
from prompt_refiner import PromptRefiner, OpenAIModel, evaluate_output
# or: from prompt_refiner import AnthropicModel

import os
os.environ["OPENAI_API_KEY"] = "sk-..."

model = OpenAIModel(model="gpt-4o-mini")
refiner = PromptRefiner(target_model=model, evaluate_fn=evaluate_output)

result = refiner.refine(
    raw_prompt="Summarize this article.",
    task_type="summarization",
    criteria={
        "max_words": 120,
        "required_keywords": ["summary", "key", "point"],
        "reference_text": "A concise summary covering the key points.",
    },
    quality_threshold=0.8,
    max_iterations=5,
)

print(result.refined_prompt)
print(result.final_output)
print(result.quality_score)
```

## Algorithm summary

**Inputs:** `raw_prompt`, `task_type`, `target_model`, `few_shot_examples`
(optional), `quality_threshold`, `max_iterations`, `criteria`.

**Outputs:** `refined_prompt`, `final_output`, `quality_score`,
`iteration_log`.

**Conditions:** five independent checks gate whether each prompt-engineering
element (role, reasoning, examples, format, specificity) is added —
each only fires if that element isn't already present in the prompt.

**Loops:**
- `WHILE` loop drives refinement of a single prompt until it passes the
  quality bar or `max_iterations` is reached.
- `FOR` loop (`validate_on_dataset`) batch-runs the algorithm over every
  entry in the benchmark dataset.

**Required libraries:** `scikit-learn` (TF-IDF + cosine similarity for
semantic scoring), `re`, `json`, `time` (standard library). Real-model
mode optionally uses `openai` or `anthropic`.
