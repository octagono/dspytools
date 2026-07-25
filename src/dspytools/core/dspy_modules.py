"""DSPy modules for quality scoring and diagnostic feedback.

Replaces heuristic Python functions with learned DSPy modules.
These modules use ChainOfThought for step-by-step reasoning about content quality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dspy
else:
    from dspytools.core._dspy import dspy


class ContentQualitySignature(dspy.Signature):
    """Score content quality based on structure, format, and completeness.

    Analyze the content for:
    - Markdown structure (headings, sections, lists, bold)
    - Code blocks presence
    - Length adequacy (not too short, not too long)
    - Overall readability and completeness

    Return a score from 0.0 to 1.0 where:
    - 0.0 = Poor quality (empty, malformed, or missing structure)
    - 0.5 = Average quality (some structure but incomplete)
    - 1.0 = Excellent quality (well-structured, comprehensive, readable)
    """

    content: str = dspy.InputField(desc="Content to evaluate for quality")
    target_format: str = dspy.InputField(
        desc="Expected format: 'markdown', 'text', or 'code'", default="markdown"
    )

    score: float = dspy.OutputField(desc="Quality score from 0.0 to 1.0")
    reasoning: str = dspy.OutputField(desc="Brief explanation of the score")


class FeedbackSignature(dspy.Signature):
    """Generate rich textual feedback from prediction and optional gold answer.

    Combine quality scoring with diagnostic feedback to provide comprehensive
    evaluation of the prediction content.

    Return both a quality score and detailed feedback explaining what's good
    and what needs improvement.
    """

    prediction: str = dspy.InputField(desc="Content to evaluate")
    gold: str = dspy.InputField(
        desc="Expected content for comparison (optional)", default=""
    )

    score: float = dspy.OutputField(desc="Quality score from 0.0 to 1.0")
    feedback: str = dspy.OutputField(desc="Detailed diagnostic feedback")


class ContentQualityScorer(dspy.Module):
    """DSPy module for scoring content quality.

    Uses ChainOfThought to reason about content structure and quality.
    Falls back to heuristic scoring if LM is unavailable.
    """

    def __init__(self):
        super().__init__()
        self.scorer = dspy.ChainOfThought(ContentQualitySignature)

    def forward(self, content: str, target_format: str = "markdown") -> dspy.Prediction:
        """Score content quality.

        Args:
            content: Content to evaluate
            target_format: Expected format ('markdown', 'text', 'code')

        Returns:
            Prediction with 'score' and 'reasoning' fields
        """
        if not content or len(content.strip()) < 50:
            return dspy.Prediction(score=0.0, reasoning="Content too short or empty")

        return self.scorer(content=content, target_format=target_format)


class RichFeedbackGenerator(dspy.Module):
    """DSPy module for generating comprehensive feedback.

    Combines quality scoring with diagnostic feedback in a single call.
    Most efficient for GEPA optimization where both score and feedback are needed.
    """

    def __init__(self):
        super().__init__()
        self.feedback_gen = dspy.ChainOfThought(FeedbackSignature)

    def forward(self, prediction: str, gold: str | None = None) -> dspy.Prediction:
        """Generate rich feedback with score and diagnostics.

        Args:
            prediction: Content to evaluate
            gold: Expected content for comparison (optional)

        Returns:
            Prediction with 'score' and 'feedback' fields
        """
        return self.feedback_gen(prediction=prediction, gold=gold or "")


class TaskProfileSignature(dspy.Signature):
    """Classify a task into domain and complexity for optimizer selection.

    Analyze the task description and metadata to determine:
    - Domain: documentation, classification, generation, reasoning, general
    - Complexity: simple, moderate, complex

    This profile guides the SelfEvolveEngine in selecting the best optimizer.
    """

    description: str = dspy.InputField(desc="Task description or prompt")
    field_count: int = dspy.InputField(desc="Number of input/output fields")
    data_size: int = dspy.InputField(desc="Number of training examples")

    domain: str = dspy.OutputField(
        desc="Task domain: documentation, classification, generation, reasoning, or general"
    )
    complexity: str = dspy.OutputField(desc="Complexity: simple, moderate, or complex")


class TaskProfileModule(dspy.Module):
    """Compilable task profiler — replaces keyword matching in MorphologyTracker.

    Compile: dspytools compile bootstrap-few-shot TaskProfileModule trainset.json
    """

    def __init__(self):
        super().__init__()
        self.profiler = dspy.ChainOfThought(TaskProfileSignature)

    def forward(
        self, description: str, field_count: int, data_size: int
    ) -> dspy.Prediction:
        return self.profiler(
            description=description,
            field_count=field_count,
            data_size=data_size,
        )


# Module-level singletons for reuse
_content_scorer: ContentQualityScorer | None = None
_rich_feedback_generator: RichFeedbackGenerator | None = None
_task_profiler: TaskProfileModule | None = None


def get_content_scorer() -> ContentQualityScorer:
    """Get or create the global content quality scorer."""
    global _content_scorer
    if _content_scorer is None:
        _content_scorer = ContentQualityScorer()
    return _content_scorer


def get_rich_feedback_generator() -> RichFeedbackGenerator:
    """Get or create the global rich feedback generator."""
    global _rich_feedback_generator
    if _rich_feedback_generator is None:
        _rich_feedback_generator = RichFeedbackGenerator()
    return _rich_feedback_generator


def get_task_profiler() -> TaskProfileModule:
    """Get or create the global task profiler."""
    global _task_profiler
    if _task_profiler is None:
        _task_profiler = TaskProfileModule()
    return _task_profiler
