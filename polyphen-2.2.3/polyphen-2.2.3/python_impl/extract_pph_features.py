"""
Extract features from PolyPhen output into a CSV for training.

Usage:
  .\.venv\Scripts\python.exe python_impl\extract_pph_features.py --input polyphen_output.tsv --out features.csv

The script supports two modes:
- TSV with header: it will map common column names (Psic1, Psic2, Nobs, PsicD, BLOSUM, prediction, truth).
- Free-form lines: you can specify regex patterns via --pattern (advanced).

If your PolyPhen output format differs, run the script on a small sample and I'll adapt it.
"""

import argparse
import csv
import re
from collections import OrderedDict

COMMON_COLS = ['acc', 'pos', 'aa1', 'aa2', 'Psic1', 'Psic2', 'Nobs', 'PsicD', 'BLOSUM', 'prediction', 'score', 'truth']


def detect_header(header):
    cols = {c: i for i, c in enumerate(header)}
    mapping = {}
    for name in COMMON_COLS:
        if name in cols:
            mapping[name] = cols[name]
    return mapping


def parse_tsv(inpath, outpath, truth_col=None):
    with open(inpath, newline='') as inf, open(outpath, 'w', newline='') as outf:
        reader = csv.reader(inf, delimiter='\t')
        header = next(reader)
        mapping = detect_header(header)
        # If truth_col provided as name, override mapping
        if truth_col:
            if truth_col in mapping:
                mapping['truth'] = mapping[truth_col]
            else:
                print(f'Warning: truth column {truth_col} not found in header')
        # Build output header
        out_cols = [c for c in COMMON_COLS]
        writer = csv.writer(outf)
        writer.writerow(out_cols)
        for r in reader:
            out_row = []
            for col in out_cols:
                if col in mapping and mapping[col] < len(r):
                    out_row.append(r[mapping[col]])
                else:
                    out_row.append('')
            writer.writerow(out_row)


def parse_with_regex(inpath, outpath, pattern):
    prog = re.compile(pattern)
    keys = prog.groupindex.keys() if prog.groupindex else []
    with open(inpath) as inf, open(outpath, 'w', newline='') as outf:
        writer = csv.writer(outf)
        # header: common cols plus any named groups
        out_cols = COMMON_COLS + list(keys)
        writer.writerow(out_cols)
        for line in inf:
            m = prog.search(line)
            out_row = [''] * len(out_cols)
            if not m:
                writer.writerow(out_row)
                continue
            g = m.groupdict()
            # fill known common cols if present
            for i, col in enumerate(out_cols):
                if col in g:
                    out_row[i] = g[col]
            writer.writerow(out_row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--out', '-o', required=True)
    p.add_argument('--mode', choices=['tsv', 'regex'], default='tsv')
    p.add_argument('--pattern', help='Regex pattern with named groups (for mode=regex)')
    p.add_argument('--truth-col', help='Header name for truth label (if different)')
    args = p.parse_args()

    if args.mode == 'tsv':
        parse_tsv(args.input, args.out, truth_col=args.truth_col)
    else:
        if not args.pattern:
            print('Regex mode requires --pattern')
            return
        parse_with_regex(args.input, args.out, args.pattern)

if __name__ == '__main__':
    main()
