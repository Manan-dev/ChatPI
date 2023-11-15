import os
import glob


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


def get_similarity_score(sentence1: str, sentence2: str):
    import spacy
    from spacy.cli import download

    file = "en_core_web_md"

    if not spacy.util.is_package(file):
        download(file)
    nlp = spacy.load(file)

    # Process the sentences
    doc1 = nlp(sentence1)
    doc2 = nlp(sentence2)

    # Compute the similarity score
    similarity_score = doc1.similarity(doc2)

    return similarity_score
