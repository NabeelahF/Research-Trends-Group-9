#!/usr/bin/env python3
"""Fetch UniProt FASTA for accessions found in a prediction file.

Saves sequences to `scratch/<ACC>.seq` with a single-line FASTA header.
Usage:
  python python_impl/fetch_uniprot_sequences.py sets/test.humdiv.output
"""
import sys
import time
import urllib.request
#!/usr/bin/env python3
"""Fetch UniProt FASTA for accessions found in a prediction file.

Saves sequences to `scratch/<ACC>.seq` with a single-line FASTA header.
Usage:
  python python_impl/fetch_uniprot_sequences.py sets/test.humdiv.output
"""
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


def extract_accs(pred_file):
    accs = []
    with open(pred_file, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if i == 0 and line.startswith('#'):
                continue
            parts = line.strip().split()
            if not parts:
                continue
            acc = parts[5] if len(parts) > 5 else parts[0]
            accs.append(acc)
    # unique preserve order
    seen = set()
    out = []
    for a in accs:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def fetch_uniprot_fasta(acc):
    urls = [
        f"https://rest.uniprot.org/uniprotkb/{acc}.fasta",
        f"https://www.uniprot.org/uniprot/{acc}.fasta",
    ]
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                data = r.read().decode('utf-8')
                if data and data.startswith('>'):
                    return data
        except Exception:
            continue
    return None


def save_seq_file(acc, fasta_text, out_dir):
    lines = fasta_text.strip().splitlines()
    seq = ''.join(lines[1:]) if len(lines) > 1 else ''
    if not seq:
        return False
    out_path = Path(out_dir) / f"{acc}.seq"
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(f">{acc}\n")
        fh.write(seq + '\n')
    return True


def main():
    if len(sys.argv) < 2:
        print('Usage: python fetch_uniprot_sequences.py <predictions_file>')
        sys.exit(2)
    pred = sys.argv[1]
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / 'scratch'
    out_dir.mkdir(parents=True, exist_ok=True)
    accs = extract_accs(pred)
    print(f'Accession count: {len(accs)}')
    succeeded = []
    failed = []
    for acc in accs:
        out_path = out_dir / f"{acc}.seq"
        if out_path.exists() and out_path.stat().st_size > 10:
            print('Skipping existing', acc)
            succeeded.append(acc)
            continue
        fasta = fetch_uniprot_fasta(acc)
        if fasta:
            ok = save_seq_file(acc, fasta, out_dir)
            if ok:
                print('Fetched', acc)
                succeeded.append(acc)
            else:
                print('Empty sequence for', acc)
                failed.append(acc)
        else:
            print('Failed', acc)
            failed.append(acc)
        time.sleep(0.1)
    print('\nSummary:')
    print('  succeeded:', len(succeeded))
    print('  failed:   ', len(failed))


if __name__ == '__main__':
    main()
import sys
