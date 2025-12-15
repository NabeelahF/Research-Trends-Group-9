#!/usr/bin/env python3
"""
Convert `sets/test.pph.output` to ARFF, run WEKA NaiveBayes with provided model, and save predictions.

Outputs:
 - experiments/weka_test.arff
 - experiments/weka_weka_output.txt
 - experiments/weka_preds.tsv (predicted\tprob)
"""
import os
import sys
import subprocess

# Ensure repo root is on sys.path so local `python_impl` package can be imported
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from python_impl.polyphen import weka as pph_weka


def main():
    import argparse
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    p = argparse.ArgumentParser()
    p.add_argument('--weka-jar', default=os.path.join(repo_root, 'tools', 'weka.jar'), help='Path to weka.jar')
    p.add_argument('--model', default=os.path.join(repo_root, 'models', 'HumDiv.UniRef100.NBd.f11.model'), help='WEKA model file')
    p.add_argument('--input', default=os.path.join(repo_root, 'sets', 'test.pph.output'), help='PPH output file')
    p.add_argument('--out-dir', default=os.path.join(repo_root, 'experiments'))
    args = p.parse_args()

    inp = args.input
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    arff = os.path.join(out_dir, 'weka_test.arff')
    weka_out = os.path.join(out_dir, 'weka_output.txt')
    preds_out = os.path.join(out_dir, 'weka_preds.tsv')

    print('Converting to ARFF...')
    n = pph_weka.to_arff(inp, arff)
    print(f'Wrote ARFF ({n} rows) ->', arff)

    # model path
    model = args.model
    if not os.path.exists(model):
        raise SystemExit('Model not found: ' + model)

    weka_jar = args.weka_jar
    if not os.path.exists(weka_jar):
        raise SystemExit('weka.jar not found at ' + weka_jar)

    cmd = ['java', '-Xmx1024m', '-cp', weka_jar, 'weka.classifiers.bayes.NaiveBayes', '-l', model, '-o', '-p', '0', '-T', arff]
    print('Running WEKA...')
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        print('WEKA failed. See combined output below:')
        print(proc.stdout)
        raise SystemExit(1)

    with open(weka_out, 'w', encoding='utf-8') as fh:
        fh.write(proc.stdout)
    print('WEKA output written to', weka_out)

    # parse predictions
    results = pph_weka.parse_weka_output(proc.stdout)
    with open(preds_out, 'w', encoding='utf-8') as fh:
        fh.write('predicted\tprob\n')
        for pred, prob in results:
            fh.write(f"{pred}\t{prob}\n")

    print('Predictions written to', preds_out)


if __name__ == '__main__':
    main()
