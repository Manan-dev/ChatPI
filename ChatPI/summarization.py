from typing import Optional
from transformers import pipeline
import torch
from pprint import pprint
from .utils import get_eval_score


def run(
    text: str,
    model: Optional[str] = None,
    verbosity: int = 1,
    **kwargs,
):
    # TODO: set some good defaults here?
    # kwargs.setdefault("min_length", 5)
    # kwargs.setdefault("max_length", 20)

    print("-" * 80)
    match verbosity:
        case 2:
            print(f"model: {model}")
            for k, v in kwargs.items():
                print(f"{k}: {v}")
            print("~" * 80)

    # Construct Pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipeline(
        "summarization",
        model=model,
        device=device,
    )

    # Run Pipeline

    text = text.strip()

    match verbosity:
        case 1 | 2:
            print(f"> {text}")

    res = pipe(
        text,
        **kwargs,
    )
    # pprint(res)

    # Get the result
    summary_text = "idk"
    if res and isinstance(res, list):
        assert len(res) == 1, "Expected only 1 result"
        summary_text = res[0].get("summary_text", "idk")

    summary_text = summary_text.strip()

    # print()
    print(f"> {summary_text}")

    return summary_text


def run_models(
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
