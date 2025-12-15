#!/usr/bin/env python3
"""Single-sequence inference using the fine-tuned logistic regression.

Run with a single sequence string using --seq, or provide a file with --input
that contains a single sequence or FASTA. Defaults to loading the fine-tuned
model at `experiments/baseline_tuning/best_model.joblib`.

Command to run predictions
python python_impl/infer_finetuned_lr.py --seq MKT... --id myProtein
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def read_single_sequence(path: Path):
    txt = path.read_text().strip()
    if not txt:
        return None
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    if not lines:
        return None
    if lines[0].startswith(">"):
        seq_parts = []
        for line in lines[1:]:
            if line.startswith(">"):
                break
            seq_parts.append(line)
        return "".join(seq_parts)
    # else take first token or whole line
    first = lines[0].split()
    return first[1] if len(first) > 1 else first[0]


def predict_single(model, seq: str):
    # Try predict_proba on raw string input
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([seq])
            if proba.ndim == 2 and proba.shape[1] >= 2:
                prob = float(proba[0, 1])
            else:
                prob = float(proba.ravel()[0])
            pred = int(model.predict([seq])[0]) if hasattr(model, "predict") else int(prob >= 0.5)
            return prob, pred
        if hasattr(model, "decision_function"):
            df = model.decision_function([seq])
            prob = float(sigmoid(df.ravel()[0]))
            pred = int(prob >= 0.5)
            return prob, pred
    except Exception:
        # fall through to trying transformed input if model is e.g. a bare estimator
        pass
    # try transforming via vectorizer if model has a named step 'vectorizer' or
    # if it's a Pipeline we rely on its internal transform; otherwise try raw
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([seq])
            prob = float(proba[0, 1]) if (proba.ndim == 2 and proba.shape[1] >= 2) else float(proba.ravel()[0])
            pred = int(model.predict([seq])[0]) if hasattr(model, "predict") else int(prob >= 0.5)
            return prob, pred
    except Exception:
        raise


def main():
    p = argparse.ArgumentParser(description="Infer single protein sequence with fine-tuned logistic regression")
    p.add_argument("--seq", help="Protein sequence string (single) to classify")
    p.add_argument("--input", help="Path to a file containing a single sequence or FASTA")
    p.add_argument("--model", default=None, help="Path to joblib model. Defaults to experiments/baseline_tuning/best_model.joblib")
    p.add_argument("--id", default="seq1", help="ID to use in output")
    args = p.parse_args()

    if not args.seq and not args.input:
        print("Provide a sequence via --seq or a file via --input", file=sys.stderr)
        p.print_help(file=sys.stderr)
        sys.exit(2)

    if args.seq:
        seq = args.seq.strip()
    else:
        seq_path = Path(args.input)
        if not seq_path.exists():
            print(f"Input file {seq_path} not found", file=sys.stderr)
            sys.exit(2)
        seq = read_single_sequence(seq_path)
        if seq is None:
            print("No sequence found in input file", file=sys.stderr)
            sys.exit(2)

    # resolve default model path inside repo
    base_dir = Path(__file__).resolve().parents[1]
    default_model = base_dir / "experiments" / "baseline_tuning" / "best_model.joblib"
    model_path = Path(args.model) if args.model else default_model
    if not model_path.exists():
        print(f"Model file {model_path} not found", file=sys.stderr)
        sys.exit(2)

    model = joblib.load(model_path)
    try:
        prob, pred = predict_single(model, seq)
    except Exception as e:
        print(f"Model prediction failed: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"sequence_id\t{args.id}")
    print(f"sequence\t{seq}")
    print(f"probability_positive\t{prob:.6f}")
    print(f"predicted_label\t{pred}")


if __name__ == "__main__":
    main()
