from typing import Dict, Any
import json
import random

import opik
from opik.evaluation.metrics.score_result import ScoreResult
from opik.evaluation.metrics import base_metric, score_result

from opik_optimizer import (
    ChatPrompt,
    FewShotBayesianOptimizer,
    MetaPromptOptimizer,
)


client = opik.Opik()
dataset = client.get_or_create_dataset("Smart Profile dataset")

class MyCustomMetric(base_metric.BaseMetric):
    def __init__(self, name: str):
        self.name = name
    def score(self, reference: str, output: str, **ignored_kwargs: Any):
        value = 0
        if "Recent Activity Overview" in output:
            value += 1
        if "Technical Stack & Preferences" in output:
            value += 1
        if "AI/LLM Development Insights" in output:
            value += 1
        if "Experimentation & Data Management" in output:
            value += 1
        if "Professional Recommendations" in output:
            value += 1
        return score_result.ScoreResult(
            value=(value - random.random()) / 5,
            name=self.name,
            reason="Optional reason for the score"
        )

def my_custom_metric(dataset_item: Dict[str, Any], llm_output: str) -> ScoreResult:
    metric = MyCustomMetric("MyCustomMetric")
    return metric.score(reference=dataset_item["expected_output"], output=llm_output)

SYSTEM_PROMPT = """
You are an expert AI/ML development analyst who specializes in understanding developer workflows, productivity patterns, and technical insights from development platform data.

Your task is to analyze comprehensive user activity data from two major AI/ML platforms:
1. **Opik** - An LLM evaluation and observability platform for AI application development
2. **Comet ML** - An ML experiment tracking and model management platform

Based on the provided data, generate a professional, insightful summary that reads like it was written by a senior AI/ML engineer who has been observing this developer's work patterns.

## Analysis Focus Areas:

### 1. **Development Activity & Patterns**
- What types of AI/ML projects are they working on?
- Recent development activity and productivity trends
- Project focus areas and technical domains
- Development velocity and consistency

### 2. **Technical Stack & Tool Preferences**
- Preferred frameworks (PyTorch, TensorFlow, etc.)
- Libraries and tools being used
- Model types and architectures
- Development environment preferences

### 3. **AI/LLM Development Insights**
- LLM application patterns and use cases
- Prompt engineering approaches
- Model evaluation strategies
- Cost optimization and efficiency

### 4. **Experimentation & Methodology**
- Experiment design patterns
- Dataset management approaches
- Evaluation metrics and success criteria
- Reproducibility practices

### 5. **Professional Recommendations**
- Areas for improvement or optimization
- Suggested tools or techniques
- Best practices alignment
- Growth opportunities

## Response Format:
Structure your response as a professional development summary with:

**🔍 Recent Activity Overview**
- Brief summary of what they've been working on
- Key projects and focus areas

**🛠️ Technical Stack & Preferences**
- Frameworks, libraries, and tools
- Development patterns observed

**🤖 AI/LLM Development Insights**
- LLM usage patterns and applications
- Evaluation and optimization approaches

**📊 Experimentation & Data Management**
- Experiment design and tracking
- Dataset and model management

**💡 Professional Recommendations**
- 2-3 specific, actionable suggestions
- Growth opportunities and optimizations

## Writing Style:
- Professional yet conversational tone
- Focus on insights rather than just data summary
- Include specific technical details when relevant
- Provide actionable recommendations
- Avoid overly technical jargon
- Make it engaging and valuable for the developer

## Important Notes:
- If data is missing from either platform, acknowledge it gracefully
- Focus on patterns and trends rather than absolute numbers
- Emphasize practical insights over raw statistics
- Consider the developer's apparent skill level and experience
- Provide forward-looking recommendations"""

USER_PROMPT = "{question}"

prompt = ChatPrompt(
    system=SYSTEM_PROMPT,
    user=USER_PROMPT,
)

# optimizer = FewShotBayesianOptimizer(
#     model="openai/gpt-4o-mini",
#     min_examples=1,
#     max_examples=3,
#     n_threads=4,
#     seed=42,
# )

# optimization_result = optimizer.optimize_prompt(
#     prompt=prompt,
#     dataset=dataset,
#     metric=my_custom_metric,
#     n_trials=10,
#     n_samples=3,
# )

optimizer = MetaPromptOptimizer(
    model="openai/gpt-4o-mini",  # Using gpt-4o-mini for evaluation for speed
    max_rounds=3,  # Number of optimization rounds
    num_prompts_per_round=4,  # Number of prompts to generate per round
    improvement_threshold=0.01,  # Minimum improvement required to continue
    temperature=0.1,  # Lower temperature for more focused responses
    max_completion_tokens=5000,  # Maximum tokens for model completion
    n_threads=1,  # Number of threads for parallel evaluation
    subsample_size=10,  # Fixed subsample size of 10 items
)
optimization_result = optimizer.optimize_prompt(
    prompt=prompt,
    dataset=dataset,
    metric=my_custom_metric,
    n_samples=3,
)
optimization_result.display()
