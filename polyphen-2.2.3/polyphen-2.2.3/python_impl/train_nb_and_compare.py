#!/usr/bin/env python3
"""
Train MultinomialNB on `sets/processed_dataset.csv` using char n-grams (1,3),
write predictions to `experiments/baseline_tuning/preds_nb.tsv`, and compare
against the tuned logistic predictions `experiments/baseline_tuning/preds_tuned.tsv`
using `python_impl/compare_models.py`.

Usage: run from repo root with the workspace Python.
"""
import os
import runpy
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, roc_auc_score


def write_preds(path, preds, probs, truths):
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('predicted\ttruth\tscore\n')
        for p, s, t in zip(preds, probs, truths):
            fh.write(f"{p}\t{t}\t{s}\n")


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    inp = os.path.join(repo_root, 'sets', 'processed_dataset.csv')
    out_dir = os.path.join(repo_root, 'experiments', 'baseline_tuning')
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(inp)
    X = df['sequence_window'].astype(str).values
    y = df['label'].astype(int).values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    vec = CountVectorizer(analyzer='char', ngram_range=(1,3))
    Xtr_v = vec.fit_transform(Xtr)
    Xte_v = vec.transform(Xte)

    nb = MultinomialNB()
    nb.fit(Xtr_v, ytr)
    pred_nb = nb.predict(Xte_v)
    prob_nb = nb.predict_proba(Xte_v)[:, 1] if hasattr(nb, 'predict_proba') else np.zeros(len(pred_nb))

    map_pred = lambda x: 'damaging' if int(x) == 1 else 'benign'
    preds_path = os.path.join(out_dir, 'preds_nb.tsv')
    write_preds(preds_path, [map_pred(x) for x in pred_nb], prob_nb.tolist(), [map_pred(x) for x in yte])

    # Run comparison against tuned predictions
    tuned_preds = os.path.join(out_dir, 'preds_tuned.tsv')
    comp_out = os.path.join(out_dir, 'comparison_nb_vs_tuned.txt')
    if not os.path.exists(tuned_preds):
        print('Tuned predictions not found:', tuned_preds)
        return

    # Call compare_models.py via subprocess to pass arguments
    import subprocess, sys
    cmd = [sys.executable, 'python_impl/compare_models.py', '--orig', preds_path, '--tuned', tuned_preds, '--out', comp_out]
    print('Running:', ' '.join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print('compare_models failed:', res.stderr)
        return

    print('NB predictions:', preds_path)
    print('Comparison written to:', comp_out)


if __name__ == '__main__':
    main()
