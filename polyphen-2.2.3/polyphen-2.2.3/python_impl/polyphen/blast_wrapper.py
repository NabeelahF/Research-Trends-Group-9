import subprocess
import os

def run_blastp(query_fasta, db_path, out_path, blastp_path='blastp', evalue=0.001, num_threads=1, outfmt=6):
    """
    Run BLASTP on a query FASTA file against a protein database.
    Args:
        query_fasta (str): Path to query FASTA file.
        db_path (str): Path to BLAST database (without extension).
        out_path (str): Path to output file.
        blastp_path (str): Path to blastp executable.
        evalue (float): E-value threshold.
        num_threads (int): Number of threads to use.
        outfmt (int): Output format (default 6 = tabular).
    Returns:
        None. Output written to out_path.
    """
    cmd = [
        blastp_path,
        '-query', query_fasta,
        '-db', db_path,
        '-out', out_path,
        '-evalue', str(evalue),
        '-num_threads', str(num_threads),
        '-outfmt', str(outfmt)
    ]
    print(f"Running BLASTP: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"BLASTP failed: {result.stderr}")
        raise RuntimeError(f"BLASTP failed: {result.stderr}")
    print(f"BLASTP finished. Output: {out_path}")
