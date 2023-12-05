from typing import Optional
from transformers import pipeline
import torch
from pprint import pprint
from .utils import get_eval_score


def run_sum(
    text: str,
    model: Optional[str] = None,
    verbosity: int = 1,
    **kwargs,
):
    # Construct Pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipeline(
        "summarization",
        model=model,
        device=device,
    )

    # Run Pipeline

    text = text.strip()


    res = pipe(
        text,
        **kwargs,
    )


    # Get the result
    summary_text = "idk"
    if res and isinstance(res, list):
        assert len(res) == 1, "Expected only 1 result"
        summary_text = res[0].get("summary_text", "idk")

    summary_text = summary_text.strip()

    # print()

    return summary_text


def run_sum_models(
    text: str,
    models: list[str],
    expected_answer: str,
    metric: str = "spacy_sim",
    **kwargs,
):
    if expected_answer is None:
        expected_answer = ""

    answers = []
    scores = []

    for model in models:
        a = run(text, model)
        s = get_eval_score(a, expected_answer, metric, **kwargs)

        answers.append(a)
        scores.append(s)

    return answers, scores
