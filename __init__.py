"""
prompt_refiner
==============

An implementation of the Iterative Prompt Refinement Algorithm (IPRA):
systematically transforms a vague/under-specified draft prompt into a
well-structured, high-performing prompt for generative AI models, scoring
each candidate output and looping until a quality threshold is met.
"""

from .refiner import PromptRefiner, RefinementResult
from .models import MockGenerativeModel, OpenAIModel, AnthropicModel
from .evaluation import evaluate_output, rule_based_score, semantic_score

__all__ = [
    "PromptRefiner",
    "RefinementResult",
    "MockGenerativeModel",
    "OpenAIModel",
    "AnthropicModel",
    "evaluate_output",
    "rule_based_score",
    "semantic_score",
]

__version__ = "1.0.0"
