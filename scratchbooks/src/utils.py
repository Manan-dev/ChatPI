import os
import glob
from termcolor import cprint, colored


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


def colored_score(score: float):
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

    # Compute the similarity score
    score = doc1.similarity(doc2)

    # if score > 0.9:
    #     color = "green"
    # elif score > 0.8:
    #     color = "light_green"
    # elif score > 0.65:
    #     color = "light_yellow"
    # elif score > 0.5:
    #     color = "yellow"
    # elif score > 0.25:
    #     color = "light_red"
    # else:
    #     color = "red"
    # cprint(f"SIMILARITY: {round(score, 4)}", color)

    print(f"SIMILARITY: {colored_score(score)}")

    return score


def create_plots(
    ctx_name: str,
    scores_by_model: dict[str, list],
    scores_by_answer: dict[str, dict[str, list]],
    scores_by_question_idx: dict[str, dict[str, list]],
    tablefmt="double_grid",
    savedir="./plots",
):
    from matplotlib import pyplot as plt
    import pandas as pd
    from tabulate import tabulate

    print("#" * 80)
    print("Plotting")

    os.makedirs(savedir, exist_ok=True)

    models = list(scores_by_model.keys())
    num_questions = len(scores_by_question_idx[models[0]])

    print(f"Models: {models}")
    print(f"Questions: {num_questions}")

    #############################################################################
    # scores_by_question

    # Boxplot Version
    fig = plt.figure(figsize=(10, 5))
    for i, m in enumerate(models):
        ax = fig.add_subplot(1, len(models), i + 1)
        ax.boxplot(scores_by_question_idx[m].values())
        ax.set_xticklabels(scores_by_question_idx[m].keys())
        ax.set_title(m)
        ax.set_ylabel("Evaluation Score")
        ax.set_xlabel("Question Index")

    fig.suptitle(f"QA Score by Question - (CTX: {ctx_name})")
    plt.show()
    fig.savefig(os.path.join(savedir, f"{ctx_name}.scores_by_question.png"))
    plt.close(fig)

    # Table Version
    headers = ["Q Idx", "Model", "Score"]
    table = []
    for model, questions in scores_by_question_idx.items():
        for i, scores in questions.items():
            # mean = sum(scores) / len(scores)
            # table.append([i, model, mean])
            cmean = colored_score(sum(scores) / len(scores))
            table.append([i, model, cmean])

    # sort by question index
    table = sorted(table, key=lambda x: x[0])

    print(
        tabulate(
            table,
            headers=headers,
            tablefmt=tablefmt,
        )
    )

    #############################################################################
    # scores_by_answer
    fig = plt.figure(figsize=(10, 5))

    # Boxplot Version
    expected_answers = list(scores_by_answer[models[0]].keys())
    for i, m in enumerate(models):
        ax = fig.add_subplot(1, len(models), i + 1)
        ax.boxplot(scores_by_answer[m].values())
        # ax.set_xticklabels(scores_by_answer[m].keys())
        ax.set_xticklabels(range(len(expected_answers)))

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
    for model, answers in scores_by_answer.items():
        for i, a in enumerate(answers.keys()):
            scores = answers[a]
            # mean = sum(scores) / len(scores)
            # table.append([i, a, min(scores), mean, max(scores), model])
            cmin = colored_score(min(scores))
            cmean = colored_score(sum(scores) / len(scores))
            cmax = colored_score(max(scores))
            table.append([model, i, a, cmin, cmean, cmax])

    # sort by answer index
    table = sorted(table, key=lambda x: x[1])

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
    ax.set_xticklabels(scores_by_model.keys())
    ax.set_title(f"QA Score by Model - {num_questions} Q's each (CTX: {ctx_name})")
    ax.set_ylabel("Evaluation Score")
    ax.set_xlabel("Model ID")
    plt.show()
    fig.savefig(os.path.join(savedir, f"{ctx_name}.scores_by_model.png"))
    plt.close(fig)

    # Table Version
    headers = ["Model", "Min", "Mean", "Max"]
    table = []
    for i, m in enumerate(scores_by_model.keys()):
        scores = scores_by_model[m]
        # table.append([m, min(scores), mean, max(scores)])
        cmin = colored_score(min(scores))
        cmean = colored_score(sum(scores) / len(scores))
        cmax = colored_score(max(scores))
        table.append([m, cmin, cmean, cmax])

    print(
        tabulate(
            table,
            headers=headers,
            tablefmt=tablefmt,
        )
    )
