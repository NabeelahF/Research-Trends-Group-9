from Bio import Align
from Bio.Align import substitution_matrices
import numpy as np

# Constants from C++ (constants.hpp) translated to Python/BioPython
# But BioPython has its own. We should try to match SIFT4G parameters.

def create_aligner(algorithm="SW", gap_open=10, gap_extend=1, matrix_name="BLOSUM62"):
    aligner = Align.PairwiseAligner()
    
    # Mode
    if algorithm == "SW":
        aligner.mode = 'local'
    elif algorithm == "NW":
        aligner.mode = 'global'
    else:
        # Default to local if unknown or "HW"/"OV" (Semiglobal/Overlap not directly standard names in simple config)
        # HW usually means semiglobal. BioPython supports this via mode='global' + specific wildcard penalties?
        # For now, implementing SW and NW.
        aligner.mode = 'local'

    # Matrix
    try:
        aligner.substitution_matrix = substitution_matrices.load(matrix_name)
    except Exception:
        # Fallback or error if matrix name specific to SIFT4G (e.g. BLOSUM_62 vs BLOSUM62)
        # BioPython uses "BLOSUM62" (no underscore usually)
        if matrix_name == "BLOSUM_62":
             aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        else:
             raise ValueError(f"Unknown matrix: {matrix_name}")

    # Gap penalties
    # BioPython uses negative numbers for penalties usually, but check API.
    # aligner.open_gap_score = -10
    # aligner.extend_gap_score = -0.5
    aligner.open_gap_score = -gap_open
    aligner.extend_gap_score = -gap_extend
    
    return aligner

def align_sequences(aligner, seq1, seq2):
    """
    Aligns two sequences.
    Returns the best alignment object (score, alignment strings).
    """
    alignments = aligner.align(seq1, seq2)
    if not alignments:
        return None
    return alignments[0]

# Support for specific score calculation if needed separately
