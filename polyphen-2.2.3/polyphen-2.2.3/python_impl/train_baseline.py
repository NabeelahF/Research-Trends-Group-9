"""
Baseline classifier for sequence-window dataset.
Reads `sets/processed_dataset.csv` (columns: sequence_window,label)
Trains a logistic regression on character n-grams and prints metrics.

Usage:
  .\.venv\Scripts\python.exe python_impl\train_baseline.py --input sets/processed_dataset.csv --out results_baseline.txt

Requires: scikit-learn
  .\.venv\Scripts\python.exe -m pip install scikit-learn
"""

import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--out', default='results_baseline.txt')
    p.add_argument('--test-size', type=float, default=0.2)
    p.add_argument('--random-state', type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    if 'sequence_window' not in df.columns or 'label' not in df.columns:
        print('Expected columns: sequence_window,label')
        print('Columns found:', df.columns.tolist())
        sys.exit(1)

    X = df['sequence_window'].astype(str).values
    y = df['label'].astype(int).values

    # vectorize with character n-grams
    vec = CountVectorizer(analyzer='char', ngram_range=(1,3))
    Xv = vec.fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(Xv, y, test_size=args.test_size, random_state=args.random_state, stratify=y)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)

    ypred = clf.predict(Xte)
    yprob = clf.predict_proba(Xte)[:, 1] if hasattr(clf, 'predict_proba') else None

    report = classification_report(yte, ypred, digits=4)
    cm = confusion_matrix(yte, ypred)

    with open(args.out, 'w') as fh:
        fh.write('Classification report:\n')
        fh.write(report + '\n')
        fh.write('Confusion matrix:\n')
        fh.write('\n'.join(['\t'.join(map(str, row)) for row in cm]) + '\n')
        if yprob is not None and len(np.unique(yte)) == 2:
            try:
                auc = roc_auc_score(yte, yprob)
                fh.write(f'ROC AUC: {auc:.4f}\n')
            except Exception as e:
                fh.write(f'ROC AUC not computed: {e}\n')

    print('Done. Results written to', args.out)

if __name__ == '__main__':
    main()
