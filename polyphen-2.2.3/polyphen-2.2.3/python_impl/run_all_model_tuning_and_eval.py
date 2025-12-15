#!/usr/bin/env python3
"""
Train baseline and tuned versions of MultinomialNB and LogisticRegression
on `sets/processed_dataset.csv` using a fixed train/test split, save predictions,
and evaluate all four models (accuracy, AUC, MCC, reports).

Produces prediction TSVs and calls `evaluate_preds_metrics.py --all` to write
`experiments/baseline_tuning/metrics_summary.json` and a TXT summary.

Usage:
  C:/Python310/python.exe python_impl/run_all_model_tuning_and_eval.py
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


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

    # Baseline vectorizer
    vec = CountVectorizer(analyzer='char', ngram_range=(1,3), min_df=1)
    Xtr_v = vec.fit_transform(Xtr)
    Xte_v = vec.transform(Xte)

    # 1) Baseline Logistic
    log_base = LogisticRegression(max_iter=1000)
    log_base.fit(Xtr_v, ytr)
    pred_log_base = log_base.predict(Xte_v)
    prob_log_base = log_base.predict_proba(Xte_v)[:, 1] if hasattr(log_base, 'predict_proba') else np.zeros(len(pred_log_base))
    write_preds(os.path.join(out_dir, 'preds_logistic_baseline.tsv'), ['damaging' if int(p)==1 else 'benign' for p in pred_log_base], prob_log_base.tolist(), ['damaging' if int(t)==1 else 'benign' for t in yte])
    joblib.dump({'vec': vec, 'clf': log_base}, os.path.join(out_dir, 'logistic_baseline.joblib'))

    # 2) Tuned Logistic (GridSearch)
    pipe_log = Pipeline([('vect', CountVectorizer(analyzer='char')), ('clf', LogisticRegression(max_iter=1000))])
    param_grid_log = {
        'vect__ngram_range': [(1,2),(1,3),(1,4)],
        'vect__min_df': [1,2],
        'clf__C': [0.01, 0.1, 1.0, 10.0]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs_log = GridSearchCV(pipe_log, param_grid_log, scoring='roc_auc', cv=cv, n_jobs=2, verbose=1)
    gs_log.fit(Xtr, ytr)
    best_log = gs_log.best_estimator_
    joblib.dump(best_log, os.path.join(out_dir, 'logistic_tuned.joblib'))
    pred_log_tuned = best_log.predict(Xte)
    prob_log_tuned = best_log.predict_proba(Xte)[:,1] if hasattr(best_log, 'predict_proba') else np.zeros(len(pred_log_tuned))
    write_preds(os.path.join(out_dir, 'preds_logistic_tuned.tsv'), ['damaging' if int(p)==1 else 'benign' for p in pred_log_tuned], prob_log_tuned.tolist(), ['damaging' if int(t)==1 else 'benign' for t in yte])

    # 3) Baseline Multinomial NB
    nb_base = MultinomialNB()
    nb_base.fit(Xtr_v, ytr)
    pred_nb_base = nb_base.predict(Xte_v)
    prob_nb_base = nb_base.predict_proba(Xte_v)[:,1] if hasattr(nb_base, 'predict_proba') else np.zeros(len(pred_nb_base))
    write_preds(os.path.join(out_dir, 'preds_nb_baseline.tsv'), ['damaging' if int(p)==1 else 'benign' for p in pred_nb_base], prob_nb_base.tolist(), ['damaging' if int(t)==1 else 'benign' for t in yte])
    joblib.dump({'vec': vec, 'clf': nb_base}, os.path.join(out_dir, 'nb_baseline.joblib'))

    # 4) Tuned Multinomial NB (grid over alpha and vectorizer)
    pipe_nb = Pipeline([('vect', CountVectorizer(analyzer='char')), ('clf', MultinomialNB())])
    param_grid_nb = {
        'vect__ngram_range': [(1,2),(1,3),(1,4)],
        'vect__min_df': [1,2],
        'clf__alpha': [1e-3, 1e-2, 1e-1, 1.0]
    }
    gs_nb = GridSearchCV(pipe_nb, param_grid_nb, scoring='roc_auc', cv=cv, n_jobs=2, verbose=1)
    gs_nb.fit(Xtr, ytr)
    best_nb = gs_nb.best_estimator_
    joblib.dump(best_nb, os.path.join(out_dir, 'nb_tuned.joblib'))
    pred_nb_tuned = best_nb.predict(Xte)
    prob_nb_tuned = best_nb.predict_proba(Xte)[:,1] if hasattr(best_nb, 'predict_proba') else np.zeros(len(pred_nb_tuned))
    write_preds(os.path.join(out_dir, 'preds_nb_tuned.tsv'), ['damaging' if int(p)==1 else 'benign' for p in pred_nb_tuned], prob_nb_tuned.tolist(), ['damaging' if int(t)==1 else 'benign' for t in yte])

    # Rename files to match evaluator expectations (overwrite existing preds)
    os.replace(os.path.join(out_dir, 'preds_logistic_baseline.tsv'), os.path.join(out_dir, 'preds_baseline.tsv'))
    os.replace(os.path.join(out_dir, 'preds_logistic_tuned.tsv'), os.path.join(out_dir, 'preds_tuned.tsv'))
    os.replace(os.path.join(out_dir, 'preds_nb_baseline.tsv'), os.path.join(out_dir, 'preds_nb.tsv'))
    os.replace(os.path.join(out_dir, 'preds_nb_tuned.tsv'), os.path.join(out_dir, 'preds_nb_tuned.tsv'))

    print('Trained and saved models and predictions in', out_dir)
    print('Now run:')
    print('  C:/Python310/python.exe python_impl/evaluate_preds_metrics.py --all')


if __name__ == '__main__':
    main()
