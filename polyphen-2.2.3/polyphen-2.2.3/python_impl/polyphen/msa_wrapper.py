import subprocess
import os

def run_mafft(input_fasta, output_fasta, mafft_path='mafft', num_threads=1, extra_args=None):
    """
    Run MAFFT to generate a multiple sequence alignment (MSA).
    Args:
        input_fasta (str): Path to input FASTA file (sequences to align).
        output_fasta (str): Path to output FASTA file (aligned sequences).
        mafft_path (str): Path to MAFFT executable.
        num_threads (int): Number of threads to use.
        extra_args (list): Additional command-line arguments for MAFFT.
    Returns:
        None. Output written to output_fasta.
    """
    cmd = [mafft_path, '--thread', str(num_threads)]
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(input_fasta)
    print(f"Running MAFFT: {' '.join(cmd)}")
    with open(output_fasta, 'w') as out_f:
        result = subprocess.run(cmd, stdout=out_f, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"MAFFT failed: {result.stderr}")
        raise RuntimeError(f"MAFFT failed: {result.stderr}")
    print(f"MAFFT finished. Output: {output_fasta}")
