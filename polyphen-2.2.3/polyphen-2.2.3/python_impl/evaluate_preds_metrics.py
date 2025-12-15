#!/usr/bin/env python3
"""
Compute metrics (Accuracy, ROC AUC, MCC, classification report, confusion matrix)
from prediction TSV files with columns: predicted\ttruth\tscore

Usage:
  C:/Python310/python.exe python_impl/evaluate_preds_metrics.py --preds experiments/baseline_tuning/preds_baseline.tsv --name baseline
  C:/Python310/python.exe python_impl/evaluate_preds_metrics.py --all
"""
import argparse
import os
import csv
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, matthews_corrcoef)


def read_preds(path):
    preds = []
    truths = []
    scores = []
    with open(path, newline='', encoding='utf-8') as fh:
        reader = csv.reader(fh, delimiter='\t')
        header = next(reader)
        # expect header: predicted, truth, score
        for r in reader:
            if len(r) < 2:
                continue
            pred = r[0].strip()
            truth = r[1].strip()
            score = None
            if len(r) >= 3:
                try:
                    score = float(r[2])
                except Exception:
                    score = None
            preds.append(pred)
            truths.append(truth)
            scores.append(score)
    return preds, truths, scores


def to_bin_label(x):
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ('1','true','t','damaging','d','positive','pos'):
        return 1
    if s in ('0','false','f','benign','neutral','neg','negative'):
        return 0
    # try numeric
    try:
        v = float(s)
        return 1 if v >= 0.5 else 0
    except Exception:
        return None


def evaluate(path, name=None):
    preds, truths, scores = read_preds(path)
    y_pred = [to_bin_label(p) for p in preds]
    y_true = [to_bin_label(t) for t in truths]

    # filter out unknowns
    valid_idx = [i for i, y in enumerate(y_true) if y is not None]
    if not valid_idx:
        print(f'No valid truth labels in {path}')
        return None

    y_true_arr = np.array([y_true[i] for i in valid_idx])
    y_pred_arr = np.array([y_pred[i] if y_pred[i] is not None else 0 for i in valid_idx])
    scores_arr = np.array([scores[i] for i in valid_idx])

    acc = accuracy_score(y_true_arr, y_pred_arr)
    report = classification_report(y_true_arr, y_pred_arr, digits=4, zero_division=0)
    cm = confusion_matrix(y_true_arr, y_pred_arr)
    mcc = matthews_corrcoef(y_true_arr, y_pred_arr)

    auc = None
    if np.any(~np.equal(scores_arr, None)):
        # need to filter scores that are not None
        valid_scores = ~np.isnan(scores_arr.astype(float))
        if valid_scores.sum() >= 10 and len(np.unique(y_true_arr[valid_scores])) == 2:
            try:
                auc = roc_auc_score(y_true_arr[valid_scores], scores_arr[valid_scores].astype(float))
            except Exception:
                auc = None

    out = {
        'path': path,
        'name': name or os.path.basename(path),
        'accuracy': acc,
        'mcc': mcc,
        'auc': auc,
        'report': report,
        'confusion_matrix': cm.tolist()
    }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preds', help='Prediction TSV file (predicted\ttruth\tscore)')
    p.add_argument('--name', help='Name for the preds file')
    p.add_argument('--all', action='store_true', help='Evaluate baseline, tuned, and nb preds in experiments/baseline_tuning')
    args = p.parse_args()

    results = []
    if args.all:
        base = 'experiments/baseline_tuning'
        files = [
            (os.path.join(base, 'preds_baseline.tsv'), 'logistic_baseline'),
            (os.path.join(base, 'preds_tuned.tsv'), 'logistic_tuned'),
            (os.path.join(base, 'preds_nb.tsv'), 'nb_multinomial')
        ]
        for fp, name in files:
            if os.path.exists(fp):
                res = evaluate(fp, name)
                if res:
                    results.append(res)
            else:
                print('Missing file:', fp)
    else:
        if not args.preds:
            print('Specify --preds or --all')
            return
        results.append(evaluate(args.preds, args.name))

    # print summary
    for r in results:
        print('---')
        print('Model:', r['name'])
        print('File:', r['path'])
        print(f"Accuracy: {r['accuracy']:.4f}")
        if r['auc'] is not None:
            print(f"ROC AUC: {r['auc']:.4f}")
        else:
            print('ROC AUC: n/a (no scores)')
        print(f"MCC: {r['mcc']:.4f}")
        print('Confusion matrix:', r['confusion_matrix'])
        print('\nClassification report:\n')
        print(r['report'])

    # write a combined JSON summary
    try:
        import json
        outf = os.path.join('experiments', 'baseline_tuning', 'metrics_summary.json')
        with open(outf, 'w', encoding='utf-8') as fh:
            json.dump(results, fh, indent=2)
        print('Wrote summary to', outf)
    except Exception:
        pass


if __name__ == '__main__':
    main()
