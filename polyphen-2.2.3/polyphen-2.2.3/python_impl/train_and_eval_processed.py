#!/usr/bin/env python3
"""Train Logistic Regression (default + tuned) and MultinomialNB on processed_dataset.csv
Vectorizes sequence windows (char n-grams) and evaluates models.
Saves predictions, metrics, plots and models to an outputs folder.
"""
import argparse
import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support, brier_score_loss,
                             roc_curve, precision_recall_curve, accuracy_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def evaluate_and_save(y_true, scores_dict, outdir):
    ensure_dir(outdir)
    results = {}
    # summary metrics
    for name, probs in scores_dict.items():
        auc = roc_auc_score(y_true, probs)
        pr_auc = average_precision_score(y_true, probs)
        preds = (probs >= 0.5).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average='binary')
        brier = brier_score_loss(y_true, probs)
        acc = accuracy_score(y_true, preds)
        results[name] = dict(auc=auc, pr_auc=pr_auc, precision=prec, recall=rec, f1=f1, brier=brier, acc=acc)

    # print summary
    for name, r in results.items():
        print(f"{name}: AUC={r['auc']:.4f}, PR-AUC={r['pr_auc']:.4f}, F1={r['f1']:.4f}, Brier={r['brier']:.4f}")

    # ROC plot
    plt.figure(figsize=(6,5))
    for name, probs in scores_dict.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['auc']:.3f})")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC'); plt.legend(loc='lower right')
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'roc_compare.png'), dpi=150)

    # PR plot
    plt.figure(figsize=(6,5))
    for name, probs in scores_dict.items():
        prec_vals, rec_vals, _ = precision_recall_curve(y_true, probs)
        plt.plot(rec_vals, prec_vals, label=f"{name} (PR-AUC={results[name]['pr_auc']:.3f})")
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall'); plt.legend(loc='lower left')
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'pr_compare.png'), dpi=150)

    return results


def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    df = pd.read_csv(args.csv, sep=',', nrows=args.max_rows)
    if 'sequence_window' not in df.columns or 'label' not in df.columns:
        logging.error('expected columns: sequence_window,label')
        sys.exit(1)

    X_text = df['sequence_window'].astype(str).values
    y = df['label'].astype(int).values

    # vectorizers (char n-grams similar to PolyPhen window encoding)
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1,3))
    counts = CountVectorizer(analyzer='char', ngram_range=(1,3))

    logging.info('fitting vectorizers...')
    X_tfidf = tfidf.fit_transform(X_text)
    X_counts = counts.fit_transform(X_text)

    X_train_tfidf, X_test_tfidf, X_train_counts, X_test_counts, y_train, y_test = train_test_split(
        X_tfidf, X_counts, y, test_size=args.test_size, random_state=42, stratify=y)

    ensure_dir(args.outdir)

    # Logistic Regression: default and tuned (use tfidf)
    logging.info('training default LogisticRegression...')
    lr_default = LogisticRegression(max_iter=1000, solver='saga')
    lr_default.fit(X_train_tfidf, y_train)

    logging.info('training tuned LogisticRegression with GridSearch...')
    param_grid = {'C': [0.01, 0.1, 1.0, 10.0], 'penalty': ['l2']}
    lr = LogisticRegression(max_iter=1000, solver='saga', class_weight='balanced')
    gs_lr = GridSearchCV(lr, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    gs_lr.fit(X_train_tfidf, y_train)
    best_lr = gs_lr.best_estimator_
    logging.info(f'LR best params: {gs_lr.best_params_}, CV AUC={gs_lr.best_score_:.4f}')

    # Multinomial NB (use counts)
    logging.info('training MultinomialNB with GridSearch...')
    mnb = MultinomialNB()
    gs_mnb = GridSearchCV(mnb, {'alpha': [0.1, 0.5, 1.0]}, cv=3, scoring='roc_auc', n_jobs=-1)
    gs_mnb.fit(X_train_counts, y_train)
    best_mnb = gs_mnb.best_estimator_
    logging.info(f'MNB best params: {gs_mnb.best_params_}, CV AUC={gs_mnb.best_score_:.4f}')

    # predict probabilities on test set
    logging.info('predicting on test set...')
    prob_lr_tuned = best_lr.predict_proba(X_test_tfidf)[:, 1]
    prob_lr_default = lr_default.predict_proba(X_test_tfidf)[:, 1]
    prob_nb = best_mnb.predict_proba(X_test_counts)[:, 1]

    # save predictions
    out_pred = os.path.join(args.outdir, 'predictions_processed.tsv')
    out_df = pd.DataFrame({
        'idx': np.arange(len(y_test)),
        'true': y_test,
        'score_lr_tuned': prob_lr_tuned,
        'score_lr_default': prob_lr_default,
        'score_nb': prob_nb,
    })
    out_df.to_csv(out_pred, sep='\t', index=False)
    logging.info(f'saved predictions to {out_pred}')

    # evaluate and save plots
    results = evaluate_and_save(y_test, {'LR_tuned': prob_lr_tuned, 'LR_default': prob_lr_default, 'NB': prob_nb}, args.outdir)

    # save summary
    summary_file = os.path.join(args.outdir, 'summary_metrics.tsv')
    pd.DataFrame(results).T.to_csv(summary_file, sep='\t')
    logging.info(f'saved metrics to {summary_file}')

    # save models
    try:
        joblib.dump(lr_default, os.path.join(args.outdir, 'lr_default.joblib'))
        joblib.dump(best_lr, os.path.join(args.outdir, 'lr_tuned.joblib'))
        joblib.dump(best_mnb, os.path.join(args.outdir, 'mnb.joblib'))
        logging.info('saved trained models to outdir')
    except Exception as e:
        logging.warning(f'failed to save models: {e}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Train and evaluate models on processed_dataset.csv')
    p.add_argument('csv', help='path to processed_dataset.csv')
    p.add_argument('--outdir', default='python_impl/outputs', help='output directory')
    p.add_argument('--test-size', type=float, default=0.2, help='test set fraction')
    p.add_argument('--max-rows', type=int, default=None, help='limit rows for quick tests')
    args = p.parse_args()
    main(args)
