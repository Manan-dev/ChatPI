import os
import glob
import evaluate
from termcolor import colored, cprint
import numpy as np
from termcolor import cprint, colored
from matplotlib import pyplot as plt
from tabulate import tabulate


def read_context(fname, basepath=f'{os.path.realpath(__file__)}/sections'):
    globpath = os.path.join(basepath, f"{fname}.*.md")

    files = glob.glob(globpath)
    files = sorted(files)

    for fname in files:
        fnameparts = fname.split(".")
        if not fnameparts[-2].isdigit():
            # print(f"Skipping: {fname}")
            continue
        with open(fname, "r") as f:
            text = f.read().strip()
            yield fname, text


def read_quiz(fname, basepath=f'{os.path.realpath(__file__)}/sections'):
    fname = os.path.join(basepath, f"{fname}.qa.md")
    with open(fname, "r") as f:
        text = f.read().strip()

        for qa_text in text.split("---"):
            qa = qa_text.strip().split("\n")
            qa = [l for l in qa if l.strip()]

            if len(qa) > 2:
                raise ValueError(f"Too many lines in QA:\n{qa}")

            if len(qa) < 2:
                raise ValueError(f"Too few lines in QA:\n{qa}")

            q = qa[0].strip()
            a = qa[1].strip()

            yield q, a


def cscore(score: float):
    if score > 0.9:
        color = "green"
    elif score > 0.8:
        color = "light_green"
    elif score > 0.65:
        color = "light_yellow"
    elif score > 0.5:
        color = "yellow"
    elif score > 0.25:
        color = "light_red"
    else:
        color = "red"
    return colored(f"{round(score, 4)}", color)


def get_similarity_score(prediction: str, reference: str):
    import spacy
    from spacy.cli import download

    file = "en_core_web_lg"

    if not spacy.util.is_package(file):
        download(file)
    nlp = spacy.load(file)

    # Process the sentences
    doc1 = nlp(prediction)
    doc2 = nlp(reference)

    # remove stop words and punctuation
    doc1 = [t for t in doc1 if not t.is_stop and not t.is_punct]
    doc2 = [t for t in doc2 if not t.is_stop and not t.is_punct]

    # combine into a single doc
    doc1 = nlp(" ".join([t.text for t in doc1]))
    doc2 = nlp(" ".join([t.text for t in doc2]))

    # Compute the similarity score
    score = doc1.similarity(doc2)

    return score


def get_eval_score(
    prediction: str,
    reference: str,
    metric: str,
    **kwargs,
):
    prediction = prediction.strip().lower()
    reference = reference.strip().lower()

    if metric == "spacy_sim":
        score = get_similarity_score(prediction, reference)
        return dict(spacy_sim=score)

    m = evaluate.load(metric)

    # dictionary containing the evaluation metric values
    m_dict: dict = m.compute(
        predictions=[prediction],
        references=[reference],
        **kwargs,
    )

    # iterate over the dictionary and prepend the metric name to the key
    # if not already present (e.g. "bertscore" -> "bertscore_f1")
    # m_dict = {f"{metric}_{k}": v for k, v in m_dict.items() if not k.startswith(metric)}
    m_dict_new = {}
    for k, v in m_dict.items():
        if not k.startswith(metric):
            k = f"{metric}_{k}"
        m_dict_new[k] = v

    # if any of the values are lists with just 1 element, then unpack the list
    m_dict_new = {
        k: v[0] if isinstance(v, list) and len(v) == 1 else v
        for k, v in m_dict_new.items()
    }

    assert len(m_dict_new) > 0, f"Metric: {metric} returned empty dict"

    return m_dict_new
