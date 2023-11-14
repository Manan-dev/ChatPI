import os
import glob


def read_context(fname, basepath="../sections"):
    globpath = os.path.join(basepath, f"{fname}*.txt")

    files = glob.glob(globpath)
    files = sorted(files)

    fnames = [os.path.basename(f) for f in files]
    print(f"Found: {fnames}")

    for fname in files:
        with open(fname, "r") as f:
            text = f.read().strip()
            yield text
