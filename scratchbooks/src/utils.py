import os
import glob
import numpy as np
from termcolor import cprint, colored
from matplotlib import pyplot as plt
from tabulate import tabulate


def read_context(fname, basepath="../sections"):
    globpath = os.path.join(basepath, f"{fname}.*.md")

    files = glob.glob(globpath)
    files = sorted(files)

    fnames = [os.path.basename(f) for f in files]
    print(f"Found: {fnames}")

    for fname in files:
        fnameparts = fname.split(".")
        if not fnameparts[-2].isdigit():
            # print(f"Skipping: {fname}")
            continue
        with open(fname, "r") as f:
            text = f.read().strip()
            yield fname, text


def read_qa(fname, basepath="../sections"):
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


def get_similarity_score(sentence1: str, sentence2: str):
    import spacy
    from spacy.cli import download

    sentence1 = sentence1.strip().lower()
    sentence2 = sentence2.strip().lower()

    file = "en_core_web_lg"

    if not spacy.util.is_package(file):
        download(file)
    nlp = spacy.load(file)

    # Process the sentences
    doc1 = nlp(sentence1)
    doc2 = nlp(sentence2)

    # remove stop words and punctuation
    doc1 = [t for t in doc1 if not t.is_stop and not t.is_punct]
    doc2 = [t for t in doc2 if not t.is_stop and not t.is_punct]

    # combine into a single doc
    doc1 = nlp(" ".join([t.text for t in doc1]))
    doc2 = nlp(" ".join([t.text for t in doc2]))

    # Compute the similarity score
    score = doc1.similarity(doc2)

    print(f"SIMILARITY: {cscore(score)}")

    return score


def create_plots(
    ctx_name: str,
    scores_by_model: dict[str, list],
    scores_by_answer: dict[str, dict[str, list]],
    scores_by_question: dict[str, dict[str, dict]],
    tablefmt="double_grid",
    savedir="./plots",
):
    print("#" * 80)
    print("Plotting")

    os.makedirs(savedir, exist_ok=True)

    models = list(scores_by_model.keys())
    num_models = len(models)
    num_questions = len(scores_by_question[models[0]])

    print(f"Models: {models}")
    print(f"Questions: {num_questions}")

    #############################################################################
    # scores_by_question

    # Plot Version
    # First subplots are individual models
    fig = plt.figure(figsize=((num_models + 1) * 5, 5))
    all_scores = []
    for i, m in enumerate(models):
        ax = fig.add_subplot(1, len(models) + 1, i + 1)
        scores = [d["score"] for d in scores_by_question[m]]
        all_scores.append(scores)
        ax.bar(range(len(scores)), scores)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(m)
        ax.set_ylabel("Evaluation Score")
        ax.set_xlabel("Question Index")

    # Last subplot is average across all models
    ax = fig.add_subplot(1, len(models) + 1, len(models) + 1)
    # convert to numpy array and average across axis 0
    all_scores = np.array(all_scores)
    avg_scores = np.mean(all_scores, axis=0)
    ax.bar(range(len(avg_scores)), avg_scores)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Average")
    ax.set_ylabel("Evaluation Score")
    ax.set_xlabel("Question Index")

    fig.suptitle(f"QA Score by Question - (CTX: {ctx_name})")
    plt.show()
    fig.savefig(os.path.join(savedir, f"{ctx_name}.scores_by_question.png"))
    plt.close(fig)

    # Table Version
    headers = ["Q Idx", "Model", "Score", "Question", "Answer", "Expected Answer"]
    table = []
    scores = []
    for model, questions in scores_by_question.items():
        for i, data in enumerate(questions):
            score = data["score"]
            scores.append(score)
            question = data["question"]
            answer = data["answer"]
            expected_answer = data["expected_answer"]
            table.append([i, model, cscore(score), question, answer, expected_answer])

    # sort by question index
    table = sorted(table, key=lambda x: x[0])

    # last row for average across the scores
    avg_score = np.mean(scores)
    table.append(["Avg", "-", cscore(avg_score), "-", "-"])

    print(
        tabulate(
            table,
            headers=headers,
            tablefmt=tablefmt,
        )
    )

    #############################################################################
    # scores_by_answer
    fig = plt.figure(figsize=(num_models * 5, 5))

    # Boxplot Version
    expected_answers = list(scores_by_answer[models[0]].keys())
    for i, m in enumerate(models):
        ax = fig.add_subplot(1, len(models), i + 1)
        ax.boxplot(scores_by_answer[m].values())
        ax.set_xticklabels(range(len(expected_answers)))
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(m)
        ax.set_ylabel("Evaluation Score")
        ax.set_xlabel("Expected Answer Group")

    fig.suptitle(
        f"QA Score by Expected Answer - {num_questions} Q's each (CTX: {ctx_name})"
    )
    plt.show()
    fig.savefig(os.path.join(savedir, f"{ctx_name}.scores_by_answer.png"))
    plt.close(fig)

    # Table Version
    headers = ["Model", "A Idx", "Expected Answer", "Min", "Mean", "Max"]
    table = []
    min_scores, mean_scores, max_scores = [], [], []
    for model, answers in scores_by_answer.items():
        for i, a in enumerate(answers.keys()):
            scores = answers[a]
            smin = min(scores)
            smean = sum(scores) / len(scores)
            smax = max(scores)
            min_scores.append(smin)
            mean_scores.append(smean)
            max_scores.append(smax)
            table.append(
                [
                    model,
                    i,
                    a,
                    cscore(smin),
                    cscore(smean),
                    cscore(smax),
                ]
            )

    # sort by answer index
    table = sorted(table, key=lambda x: x[1])

    # average across the scores (min, mean, max)
    min_avg, mean_avg, max_avg = (
        np.mean(min_scores),
        np.mean(mean_scores),
        np.mean(max_scores),
    )
    table.append(
        [
            "Avg",
            "-",
            "-",
            cscore(min_avg),
            cscore(mean_avg),
            cscore(max_avg),
        ]
    )

    print(
        tabulate(
            table,
            headers=headers,
            tablefmt=tablefmt,
        )
    )

    #############################################################################
    # scores_by_model

    # Boxplot Version
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(scores_by_model.values())
    ax.set_xticklabels(scores_by_model.keys(), rotation=10, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"QA Score by Model - {num_questions} Q's each (CTX: {ctx_name})")
    ax.set_ylabel("Evaluation Score")
    ax.set_xlabel("Model ID")
    plt.show()
    fig.savefig(os.path.join(savedir, f"{ctx_name}.scores_by_model.png"))
    plt.close(fig)

    # Table Version
    headers = ["Model", "Min", "Mean", "Max"]
    table = []
    min_scores, mean_scores, max_scores = [], [], []
    for i, m in enumerate(scores_by_model.keys()):
        scores = scores_by_model[m]
        smin = min(scores)
        smean = sum(scores) / len(scores)
        smax = max(scores)
        min_scores.append(smin)
        mean_scores.append(smean)
        max_scores.append(smax)
        table.append(
            [
                m,
                cscore(smin),
                cscore(smean),
                cscore(smax),
            ]
        )

    # average across the scores (min, mean, max)
    min_avg, mean_avg, max_avg = (
        np.mean(min_scores),
        np.mean(mean_scores),
        np.mean(max_scores),
    )
    table.append(["Avg", cscore(min_avg), cscore(mean_avg), cscore(max_avg)])

    print(
        tabulate(
            table,
            headers=headers,
            tablefmt=tablefmt,
        )
    )
