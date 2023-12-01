from typing import Optional
from transformers import pipeline
import torch
from pprint import pprint
from .utils import get_eval_score


def run(
    text: str,
    model: Optional[str] = None,
    pipeline_name: str = "translation_en_to_fr",
    verbosity: int = 1,
    **kwargs,
):
    print("-" * 100)
    match verbosity:
        case 1:
            print(f"model: {model}")
        case 2:
            print(f"model: {model}")
            for k, v in kwargs.items():
                print(f"{k}: {v}")
            print("~" * 80)

    # Construct Pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipeline(
        pipeline_name,
        model=model,
        device=device,
    )

    # Run Pipeline

    text = text.strip()

    # match verbosity:
    #     case 0:
    #         pass
    #     case 1:
    #         print(f"> {text}")

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
    models: list[tuple[str]],
    **kwargs,
):
    answers = []
    scores = []

    for model_en_to_fr, model_fr_to_en in models:
        fr = run(
            text, model=model_en_to_fr, pipeline_name="translation_en_to_fr", **kwargs
        )
        en = run(
            fr, model=model_fr_to_en, pipeline_name="translation_fr_to_en", **kwargs
        )
        s = get_eval_score(en, text, metric="spacy_sim")
        s |= get_eval_score(en, text, metric="bertscore", lang="en")
        s |= get_eval_score(en, text, metric="rouge")

        answers.append((fr, en))
        scores.append(s)

    return answers, scores
