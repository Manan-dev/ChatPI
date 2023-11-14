import os
import glob


def read_context(fname, basepath="../sections"):
    globpath = os.path.join(basepath, f"{fname}*.txt")

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


def read_questions(fname, basepath="../sections"):
    fname = os.path.join(basepath, f"{fname}.q.txt")
    with open(fname, "r") as f:
        text = f.read().strip()
        for line in text.split("\n"):
            yield line.strip()
