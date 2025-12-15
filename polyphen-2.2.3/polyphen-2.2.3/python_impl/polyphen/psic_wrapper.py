import subprocess
import os

def run_psic(alignment_file, output_file, psic_path='psic.exe', extra_args=None):
    """
    Run PSIC profile calculation on an alignment file.
    Args:
        alignment_file (str): Path to input alignment file (Clustal format).
        output_file (str): Path to output file (profile).
        psic_path (str): Path to PSIC executable.
        extra_args (list): Additional command-line arguments for PSIC.
    Returns:
        None. Output written to output_file.
    """
    cmd = [psic_path, alignment_file, output_file]
    if extra_args:
        cmd.extend(extra_args)
    print(f"Running PSIC: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"PSIC failed: {result.stderr}")
        raise RuntimeError(f"PSIC failed: {result.stderr}")
    print(f"PSIC finished. Output: {output_file}")
