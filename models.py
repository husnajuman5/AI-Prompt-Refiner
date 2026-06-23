"""
models.py
---------
Thin adapters around generative AI back-ends. Every model exposes the same
`.generate(prompt: str) -> str` interface so the refinement algorithm can
stay agnostic to which model is actually plugged in.

MockGenerativeModel requires no API key and no network access, so the whole
algorithm can be demoed / unit-tested offline. OpenAIModel and AnthropicModel
are thin wrappers you can drop a real API key into.
"""

from __future__ import annotations
import os
import random
import re
import textwrap


class BaseModel:
    """Common interface every model adapter must implement."""

    name: str = "base-model"

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class MockGenerativeModel(BaseModel):
    """
    A deterministic, offline stand-in for a real LLM.

    It does not "understand" the prompt the way a real LLM would — instead
    it inspects which prompt-engineering elements are present (role,
    examples, format instruction, reasoning instruction, constraints) and
    produces an output whose *quality* scales with how many of those
    elements are present. This lets the refinement loop be demonstrated
    and unit-tested end-to-end without any API key or internet access.
    """

    name = "mock-model-v1"

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    _INSTRUCTION_PREFIXES = (
        r"^you are",
        r"^think through",
        r"^respond in",
        r"^be specific:",
        r"^be sure to (explicitly )?mention",
        r"^input:",
        r"^output:",
    )

    def _extract_task_line(self, prompt: str) -> str:
        """Return the original task description, ignoring any lines that were
        injected by the refinement augmentations (role/reasoning/format/etc.)."""
        candidates = []
        for line in prompt.strip().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if any(re.match(pat, stripped, re.I) for pat in self._INSTRUCTION_PREFIXES):
                continue
            candidates.append(stripped)
        return candidates[-1] if candidates else prompt.strip().splitlines()[-1]

    def _extract_emphasized_keywords(self, prompt: str) -> list[str]:
        match = re.search(r"be sure to (?:explicitly )?mention:\s*(.+)", prompt, re.I)
        if not match:
            return []
        return [kw.strip(" .") for kw in match.group(1).split(",") if kw.strip()]

    def generate(self, prompt: str) -> str:
        has_role = bool(re.search(r"you are|as an? expert", prompt, re.I))
        has_examples = "Input:" in prompt and "Output:" in prompt
        has_format = bool(re.search(r"json|bullet|word limit|format", prompt, re.I))
        has_reasoning = bool(re.search(r"step by step|think through", prompt, re.I))
        has_constraints = bool(re.search(r"audience|tone|words|length", prompt, re.I))

        quality_elements = sum([has_role, has_examples, has_format, has_reasoning, has_constraints])
        task_line = self._extract_task_line(prompt)
        emphasized = self._extract_emphasized_keywords(prompt)
        emphasis_sentence = f" This explicitly covers: {', '.join(emphasized)}." if emphasized else ""

        if quality_elements == 0:
            # Sparse, generic, low-effort response
            return f"Here is something about: {task_line}{emphasis_sentence}"

        if quality_elements <= 2:
            return textwrap.dedent(f"""
                Response to: {task_line}
                This covers the main idea but lacks structure and detail.{emphasis_sentence}
            """).strip()

        # quality_elements >= 3: well-formed, structured response
        bullet_points = "\n".join(f"- Point {i+1} relevant to: {task_line}" for i in range(3))
        return textwrap.dedent(f"""
            Summary:
            A clear, well-structured response addressing: {task_line}{emphasis_sentence}

            Key Points:
            {bullet_points}

            This response follows the requested format, tone, and reasoning constraints.
        """).strip()


class OpenAIModel(BaseModel):
    """
    Wrapper around the OpenAI Python SDK.
    Requires: pip install openai, and OPENAI_API_KEY set in the environment.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        from openai import OpenAI  # imported lazily so the package works without openai installed
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.name = model

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


class AnthropicModel(BaseModel):
    """
    Wrapper around the Anthropic Python SDK.
    Requires: pip install anthropic, and ANTHROPIC_API_KEY set in the environment.
    """

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        import anthropic  # imported lazily so the package works without anthropic installed
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.name = model

    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.name,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")
