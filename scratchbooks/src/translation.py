from typing import Optional
from transformers import pipeline
import torch
from pprint import pprint
from .utils import get_similarity_score


def run(
    text: str,
    model: Optional[str] = None,
    verbosity: int = 1,
    **kwargs,
):
    print("-" * 100)
    match verbosity:
        case 2:
            print(f"model: {model}")
            for k, v in kwargs.items():
                print(f"{k}: {v}")
            print("~" * 80)

    # Construct Pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipeline(
        "translation_en_to_fr",
        model=model,
        device=device,
    )

    # Run Pipeline

    text = text.strip()

    match verbosity:
        case 0:
            pass
        case 1:
            print(f"> {text}")

    res = pipe(
        text,
        **kwargs,
    )
    # pprint(res)

    # Get the result
    translation_text = "idk"
    if res and isinstance(res, list):
        assert len(res) == 1, "Expected only 1 result"
        translation_text = res[0].get("translation_text", "idk")

    translation_text = translation_text.strip()

    # print()
    print(f"> {translation_text}")
    return translation_text


def run_models(
    text: str,
    models: list[str],
    expected_answer: str,
    **kwargs,
):
    if expected_answer is None:
        expected_answer = ""

    answers = []
    scores = []

    for model in models:
        a = run(text, model=model, **kwargs)
        s = get_similarity_score(a, expected_answer)
        answers.append(a)
        scores.append(s)

    return answers, scores
