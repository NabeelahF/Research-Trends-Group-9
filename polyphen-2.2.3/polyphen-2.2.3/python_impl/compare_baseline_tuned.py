#!/usr/bin/env python3
"""
Compare a default logistic baseline with a tuned model on the processed dataset.

Writes prediction TSVs and a short summary. Intended to be run after
`baseline_tune.py` (which saves `best_model.joblib`). If that file is absent
the script will fit a GridSearch on the training split.

Usage:
  python python_impl/compare_baseline_tuned.py --input sets/processed_dataset.csv \
      --out-dir experiments/baseline_tuning --test-size 0.2 --random-state 42 \
      --tuned experiments/baseline_tuning/best_model.joblib
"""
import argparse
import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def write_preds(path, preds, probs, truths):
    # columns: predicted\ttruth\tscore
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('predicted\ttruth\tscore\n')
        for p, s, t in zip(preds, probs, truths):
            fh.write(f"{p}\t{t}\t{s}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--out-dir', '-o', default='experiments/baseline_tuning')
    p.add_argument('--test-size', type=float, default=0.2)
    p.add_argument('--random-state', type=int, default=42)
    p.add_argument('--tuned', help='Path to tuned joblib model (optional)')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    if 'sequence_window' not in df.columns or 'label' not in df.columns:
        raise SystemExit('Input CSV must contain sequence_window and label columns')

    X = df['sequence_window'].astype(str).values
    y = df['label'].astype(int).values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size,
                                         random_state=args.random_state, stratify=y)

    vec = CountVectorizer(analyzer='char', ngram_range=(1,3))
    Xtr_v = vec.fit_transform(Xtr)
    Xte_v = vec.transform(Xte)

    # baseline (default logistic)
    baseline = LogisticRegression(max_iter=1000)
    baseline.fit(Xtr_v, ytr)
    pred_b = baseline.predict(Xte_v)
    prob_b = baseline.predict_proba(Xte_v)[:, 1] if hasattr(baseline, 'predict_proba') else np.zeros(len(pred_b))

    # tuned model: try to load joblib, otherwise perform GridSearch on training split
    tuned_model = None
    if args.tuned and os.path.exists(args.tuned):
        tuned_model = joblib.load(args.tuned)
    else:
        # run GridSearch on training split (same grid as baseline_tune.py)
        pipeline = [('vect', vec), ('clf', LogisticRegression(max_iter=1000))]
        # We'll just grid search over C and vectorizer params using a quick local pipeline
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([('vect', CountVectorizer(analyzer='char')), ('clf', LogisticRegression(max_iter=1000))])
        # Use a smaller grid / fewer folds for large datasets to keep runtime reasonable
        if len(Xtr) > 20000:
            param_grid = {
                'vect__ngram_range': [(1,3)],
                'vect__min_df': [1],
                'clf__C': [0.1, 1.0]
            }
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        else:
            param_grid = {
                'vect__ngram_range': [(1,2), (1,3)],
                'vect__min_df': [1,2],
                'clf__C': [0.1, 1.0, 10.0]
            }
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        gs = GridSearchCV(pipe, param_grid, scoring='roc_auc', cv=cv, n_jobs=2, verbose=1)
        gs.fit(Xtr, ytr)
        tuned_model = gs.best_estimator_
        # save
        try:
            joblib.dump(tuned_model, os.path.join(args.out_dir, 'best_model_from_compare.joblib'))
        except Exception:
            pass

    # ensure we have vectorizer + clf accessible
    # If tuned_model is a pipeline (vect+clf), transform using its vect
    try:
        pred_t = tuned_model.predict(Xte)
        prob_t = tuned_model.predict_proba(Xte)[:, 1] if hasattr(tuned_model, 'predict_proba') else np.zeros(len(pred_t))
    except Exception:
        # maybe tuned_model expects vector inputs (it may be a Pipeline saved with joblib)
        if hasattr(tuned_model, 'named_steps') and 'vect' in tuned_model.named_steps:
            Xte_v2 = tuned_model.named_steps['vect'].transform(Xte)
            clf = tuned_model.named_steps.get('clf', tuned_model)
            pred_t = clf.predict(Xte_v2)
            prob_t = clf.predict_proba(Xte_v2)[:, 1] if hasattr(clf, 'predict_proba') else np.zeros(len(pred_t))
        else:
            raise

    # Map numeric preds to labels compatible with compare_models/to_bin
    # use 1 -> 1/positive/damaging, 0 -> 0/benign/neutral
    map_pred = lambda x: 'damaging' if int(x) == 1 else 'benign'

    preds_dir = args.out_dir
    preds_base = os.path.join(preds_dir, 'preds_baseline.tsv')
    preds_tuned = os.path.join(preds_dir, 'preds_tuned.tsv')

    write_preds(preds_base, [map_pred(x) for x in pred_b], prob_b.tolist(), [map_pred(x) for x in yte])
    write_preds(preds_tuned, [map_pred(x) for x in pred_t], prob_t.tolist(), [map_pred(x) for x in yte])

    # small summary
    summary = []
    summary.append('Baseline classification report:')
    summary.append(classification_report(yte, pred_b, digits=4, zero_division=0))
    summary.append('\nTuned classification report:')
    summary.append(classification_report(yte, pred_t, digits=4, zero_division=0))
    try:
        auc_b = roc_auc_score(yte, prob_b)
        auc_t = roc_auc_score(yte, prob_t)
        summary.append(f'ROC AUC baseline: {auc_b:.4f} tuned: {auc_t:.4f}')
    except Exception:
        pass

    with open(os.path.join(preds_dir, 'compare_summary.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n\n'.join(summary))

    print('Wrote predictions:')
    print(' ', preds_base)
    print(' ', preds_tuned)
    print('Summary:', os.path.join(preds_dir, 'compare_summary.txt'))


if __name__ == '__main__':
    main()
