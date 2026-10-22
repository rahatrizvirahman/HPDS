#!/usr/bin/env python3
"""Exercise neighbor ties, vote ties, runtime k, and partial query blocks."""
import argparse
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", help="Also exercise the actual CUDA executable")
    args = parser.parse_args()
    (ROOT / "results").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="correctness-", dir=ROOT / "results") as scratch:
        folder = Path(scratch)
        header = "@relation ties\n@attribute feature numeric\n@attribute class numeric\n@data\n"
        train, test, output = (folder / name for name in ("train.arff", "test.arff", "predictions.txt"))
        train.write_text(header + "-1,1\n1,0\n3,2\n5,0\n")
        test.write_text(header + "0,1\n2,0\n4,2\n")
        configurations = [("serial", [])]
        if args.cuda:
            configurations += [("cuda", [str(block)]) for block in (1, 64, 128, 256)]
        for binary, extra in configurations:
            for k, expected in [(1, [1, 0, 2]), (2, [0, 0, 0]), (4, [0, 0, 0])]:
                subprocess.run([str(ROOT / binary), str(train), str(test), str(k)] + extra + [str(output)],
                               check=True, stdout=subprocess.PIPE, text=True)
                actual = [int(v) for v in output.read_text().splitlines()]
                if actual != expected:
                    raise AssertionError(f"{binary} {extra} k={k}, expected {expected}, got {actual}")
            for bad_k in ("0", "-1", "5", "abc", "3x"):
                result = subprocess.run([str(ROOT / binary), str(train), str(test), bad_k] + extra,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if result.returncode == 0:
                    raise AssertionError(f"{binary} accepted invalid k={bad_k}")
        # A test set may omit classes present in training.
        test.write_text(header + "4,0\n")
        subprocess.run([str(ROOT / "serial"), str(train), str(test), "1", str(output)],
                       check=True, stdout=subprocess.PIPE)
        if output.read_text() != "2\n":
            raise AssertionError("Prediction must remain valid when test labels omit a training class")
    print("PASS neighbor ties, vote ties, valid and invalid k, and prediction output")
    if args.cuda:
        print("PASS CUDA predictions with partial blocks")


if __name__ == "__main__":
    main()
