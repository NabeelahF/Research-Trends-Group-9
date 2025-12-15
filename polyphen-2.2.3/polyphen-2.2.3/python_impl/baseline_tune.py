"""
Baseline hyperparameter tuning for char n-gram LogisticRegression.
Saves grid search results and the best model (joblib).

Usage:
  .\.venv\Scripts\python.exe python_impl\baseline_tune.py --input sets/processed_dataset.csv --out-dir experiments/baseline_tuning

Requires: scikit-learn, pandas, joblib
  .\.venv\Scripts\python.exe -m pip install scikit-learn pandas joblib
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, roc_auc_score
import joblib


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--out-dir', '-o', default='experiments/baseline_tuning')
    p.add_argument('--cv', type=int, default=5)
    p.add_argument('--jobs', type=int, default=2)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    if 'sequence_window' not in df.columns or 'label' not in df.columns:
        raise SystemExit('Input CSV must contain sequence_window and label columns')

    X = df['sequence_window'].astype(str).values
    y = df['label'].astype(int).values

    pipeline = Pipeline([
        ('vect', CountVectorizer(analyzer='char')),
        ('clf', LogisticRegression(max_iter=1000, solver='liblinear'))
    ])

    param_grid = {
        'vect__ngram_range': [(1,2), (1,3), (1,4)],
        'vect__min_df': [1,2],
        'clf__C': [0.01, 0.1, 1.0, 10.0]
    }

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
    scorer = 'roc_auc'

    gs = GridSearchCV(pipeline, param_grid, scoring=scorer, cv=cv, n_jobs=args.jobs, verbose=2, return_train_score=True)
    gs.fit(X, y)

    # Save results
    results_df = pd.DataFrame(gs.cv_results_)
    results_csv = os.path.join(args.out_dir, 'grid_results.csv')
    results_df.to_csv(results_csv, index=False)

    best_json = os.path.join(args.out_dir, 'best_params.json')
    with open(best_json, 'w') as fh:
        json.dump({'best_params': gs.best_params_, 'best_score': gs.best_score_}, fh, indent=2)

    model_path = os.path.join(args.out_dir, 'best_model.joblib')
    joblib.dump(gs.best_estimator_, model_path)

    summary_txt = os.path.join(args.out_dir, 'summary.txt')
    with open(summary_txt, 'w') as fh:
        fh.write(f"Best score (cv={args.cv}) {gs.best_score_}\n")
        fh.write(f"Best params: {gs.best_params_}\n")
        fh.write(f"Results CSV: {results_csv}\n")
        fh.write(f"Saved model: {model_path}\n")

    print('Tuning complete. Results written to', args.out_dir)

if __name__ == '__main__':
    main()
