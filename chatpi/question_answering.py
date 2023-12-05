from typing import Optional
from transformers import pipeline
import torch
from pprint import pprint
from .utils import get_similarity_score, get_eval_score


def run_qa(
    question: str,
    context: str,
    model: Optional[str] = None,
    verbosity: int = 1,
    **kwargs,
):
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

    # display(Markdown(f"**A:** {answer} (score: {score})"))

    return answer


def run_qa_models(
    question: str,
    context: str,
    models: list[str],
    answer_true: str,
    **kwargs,
):
    if answer_true is None:
        answer_true = ""

    answer_preds = []
    score_dicts = []

    for model in models:
        answer_pred = run_qa(
            question,
            context,
            model=model,
            **kwargs,
        )

        # compare the predicted answer to the true answer
        score_dict_1 = get_eval_score(
            answer_pred,
            answer_true,
            metric="spacy_sim",
        )
        score_dict_2 = get_eval_score(
            answer_pred,
            answer_true,
            metric="bertscore",
            lang="en",
            model_type="microsoft/deberta-xlarge-mnli",
        )
        score_dict_3 = get_eval_score(
            answer_pred,
            answer_true,
            metric="rouge",
        )

        score_dict = {**score_dict_1, **score_dict_2, **score_dict_3}
        pprint(score_dict)

        answer_preds.append(answer_pred)
        score_dicts.append(score_dict)

    return answer_preds, score_dicts
