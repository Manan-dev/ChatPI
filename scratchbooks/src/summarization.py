from typing import Optional
from transformers import pipeline
import torch
from pprint import pprint


def run(
    text: str,
    model: Optional[str] = None,
    verbosity: int = 1,
    **kwargs,
):
    # TODO: set some good defaults here?
    # kwargs.setdefault("min_length", 5)
    # kwargs.setdefault("max_length", 20)

    print("=" * 100)
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
    **kwargs,
):
    for model in models:
        run(text, model, **kwargs)
