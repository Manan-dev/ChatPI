from typing import Optional
from transformers import pipeline
import torch
from pprint import pprint

def run(
    text: str,
    model: Optional[str] = None,
    **kwargs,
):
    print("=" * 100)
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


def run_models(text: str, models: list[str], **kwargs):
    for model in models:
        run(text, model=model, **kwargs)
