import os
import sys

def is_extant_path(path):
    """
    Checks if a path exists and returns code similar to C++ implementation.
    0: directory
    1: file
    2: other
    -1: does not exist
    """
    if os.path.exists(path):
        if os.path.isdir(path):
            return 0
        elif os.path.isfile(path):
            return 1
        else:
            return 2
    return -1

def create_file_name(name, path, extension):
    """
    Creates a full file path from directory, basename, and extension.
    """
    if path:
        safe_name = "".join([c if c.isalnum() or c in (' ', '.', '-', '_') else '_' for c in name])
        return os.path.join(path, safe_name + extension)
    else:
        safe_name = "".join([c if c.isalnum() or c in (' ', '.', '-', '_') else '_' for c in name])
        return safe_name + extension

def query_log(part, total):
    """
    Logs query processing progress.
    """
    percentage = 100 * part / float(total)
    sys.stderr.write(f"* processing queries: {percentage:.2f}/100.00% *\r")
    sys.stderr.flush()

def database_log(part, part_size, percentage):
    """
    Logs database processing progress.
    """
    sys.stderr.write(f"* processing database part {part} (size ~{part_size:.2f} GB): {percentage:.2f}/100.00% *\r")
    sys.stderr.flush()

from Bio import SeqIO

class Chain:
    """
    Represents a protein sequence (Chain in C++).
    """
    def __init__(self, header, sequence):
        self.header = header
        self.sequence = sequence
        # Add other fields if C++ Chain struct has them (e.g., length, id)
        self.length = len(sequence)

def read_fasta_chains(file_path):
    """
    Reads a FASTA file and returns a list of Chain objects.
    """
    chains = []
    if not os.path.exists(file_path):
         return chains
         
    for record in SeqIO.parse(file_path, "fasta"):
        chains.append(Chain(record.description, str(record.seq)))
    return chains
