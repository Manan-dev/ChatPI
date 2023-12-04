from typing import Optional
from transformers import pipeline
import torch
from pprint import pprint
from .utils import get_eval_score


def run_tr(
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


def run_tr_models(
    text: str,
    models: list[tuple[str]],
    **kwargs,
):
    translation_preds = []
    score_dicts = []

    for model_en_to_fr, model_fr_to_en in models:
        # translate original text from english to french
        text_fr = run_tr(
            text, model=model_en_to_fr, pipeline_name="translation_en_to_fr", **kwargs
        )
        # translate the french text back to english
        text_en = run_tr(
            text_fr,
            model=model_fr_to_en,
            pipeline_name="translation_fr_to_en",
            **kwargs,
        )

        # evaluate by comparing the text translated back to english to the original english text
        score_dict_1 = get_eval_score(
            text_en,
            text,
            metric="spacy_sim",
        )
        score_dict_2 = get_eval_score(
            text_en,
            text,
            metric="bertscore",
            lang="en",
            model_type="microsoft/deberta-xlarge-mnli",
        )
        score_dict_3 = get_eval_score(
            text_en,
            text,
            metric="rouge",
        )

        score_dict = {**score_dict_1, **score_dict_2, **score_dict_3}
        pprint(score_dict)

        translation_preds.append((text_fr, text_en))
        score_dicts.append(score_dict)

    return translation_preds, score_dicts
