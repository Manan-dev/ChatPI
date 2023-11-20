from typing import Optional
from transformers import pipeline
import torch
from pprint import pprint
from .utils import get_similarity_score


def run(
    question: str,
    context: str,
    model: Optional[str] = None,
    verbosity: int = 1,
    **kwargs,
):
    print("-" * 80)
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
        "question-answering",
        model=model,
        # model="deepset/roberta-base-squad2",
        device=device,
    )

    # Run Pipeline

    question = question.strip()
    context = context.strip()

    match verbosity:
        case 0:
            pass
        case 1:
            print(f"Q: {question}")
        case 2:
            print(f"C: {context}")
            print(f"Q: {question}")

    # display(Markdown(f"**Q:** {question}"))
    # display(Markdown(f"**C:** {context}"))

    res = pipe(
        question=question,
        context=context,
        **kwargs,
    )
    # pprint(res)

    answer, score = "idk", 1.0

    # Get the result
    if res and isinstance(res, dict):
        answer = res.get("answer", "idk")
        score = res.get("score", 1.0)

    answer = answer.strip()
    score = round(score, 3)

    print(f"A: {answer} (model confidence score: {round(score, 3)})")
    # display(Markdown(f"**A:** {answer} (score: {score})"))

    return answer


def run_models(
    question: str,
    context: str,
    models: list[str],
    expected_answer: str,
    **kwargs,
):
    if expected_answer is None:
        expected_answer = ""

    answers = []
    scores = []

    for model in models:
        a = run(question, context, model=model, **kwargs)
        s = get_similarity_score(a, expected_answer)

        answers.append(a)
        scores.append(s)

    return answers, scores
