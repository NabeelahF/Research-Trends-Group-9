#!/usr/bin/env python3
"""
SIFT 4G Single Sequence Predictor
Predicts whether a sequence variation is Deleterious or Tolerated using the fine-tuned SIFT4G model.
"""

import sys
import os
import argparse
import re
import copy
from Bio import SeqIO
from collections import defaultdict
import numpy as np

# Import local modules
import utils
import database
import alignment
import prediction

# Helper aliases from modules
Chain = utils.Chain
search_database = database.search_database
create_aligner = alignment.create_aligner
remove_seqs_percent_identical_to_query = prediction.remove_seqs_percent_identical_to_query
create_matrix = prediction.create_matrix
calc_sift_scores = prediction.calc_sift_scores

# Configuration Constants (Fine-tuned Model)
KMER_LENGTH = 5
DEFAULT_MAX_CANDIDATES = 20000
MAX_SEQUENCES = 800
SEQ_IDENTITY_THRESHOLD = 50
PREDICTION_CUTOFF = 0.15
ALIGNMENT_COVERAGE_THRESHOLD = 0.2
MIN_HOMOLOGS = 3

# Valid Amino Acids
VALID_AAS = "ACDEFGHIKLMNPQRSTVWY"

def sift_predictions_memory(alignment_strings, query, sequence_identity, exclude_query=False):
    """
    Run SIFT prediction and return scores directly (No file I/O).
    Returns (sift_scores_matrix, num_sequences_used)
    """
    if len(alignment_strings) > MAX_SEQUENCES - 1:
        alignment_strings[:] = alignment_strings[:MAX_SEQUENCES - 1]
    
    query_length = len(query.sequence)
    if len(alignment_strings) > 20 and sequence_identity < 100:
        remove_seqs_percent_identical_to_query(query, alignment_strings, sequence_identity)
    
    if not exclude_query:
        alignment_strings.insert(0, query)

    matrix = [[0.0]*26 for _ in range(query_length)]
    sift_scores = [[0.0]*26 for _ in range(query_length)]
    
    weights_1_temp = [1.0] * len(alignment_strings)
    aas_stored_temp = [0.0] * query_length
    create_matrix(alignment_strings, query, weights_1_temp, matrix, aas_stored_temp)
    calc_sift_scores(alignment_strings, query, matrix, sift_scores)
    
    return sift_scores, len(alignment_strings)

def predict(sequence):
    """
    Predicts the effect for the given sequence.
    """
    sequence = sequence.strip().upper()
    if not sequence:
        print("Error: Empty sequence provided.")
        return

    # Assuming 41-mer centered at mutation, so mutation is at index 20
    if len(sequence) != 41:
        print(f"Warning: Sequence length is {len(sequence)}. This model is optimized for 41-mer sequences.")
    
    mutation_pos = 20 # Center
    if len(sequence) <= mutation_pos:
        print(f"Error: Sequence too short to have a center position at index {mutation_pos}.")
        return

    query = Chain("query", sequence)
    queries = [query]
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "uniprot_sprot.fasta")
    
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}")
        return

    print(f"Searching database for homologs...")
    candidate_indices_list = search_database(queries, db_path, KMER_LENGTH, DEFAULT_MAX_CANDIDATES)
    candidate_indices = candidate_indices_list[0] if candidate_indices_list else []
    
    if not candidate_indices:
        print("No homologs found in database.")
        # Fallback prediction
        center_aa = sequence[mutation_pos]
        conservative = center_aa in "AGPST"
        result = "Tolerated" if conservative else "Deleterious"
        score = 1.0 if conservative else 0.0
        print(f"\nfallback_result: {result} (Score: {score:.4f} - No Homologs)")
        return

    print(f"Found {len(candidate_indices)} candidates. Aligning...")
    
    # Load specific records using index
    db_index = SeqIO.index(db_path, "fasta")
    db_keys = list(db_index.keys()) # This might differ if keys are not just indices...
    # database.search_database returns INDICES into the file iteration 0,1,2...
    # BUT SeqIO.index returns a dictionary where keys are record IDs.
    # WE MUST CHECK if `search_database` returns integer indices or IDs.
    # Looking at `database.py`: `db_idx` is incremented. So it returns 0-based index.
    # But `SeqIO.index` keys are strings (IDs).
    # We cannot access `db_index` by integer position efficiently without correct keys.
    # `run_finetuned_full_eval.py` lines 95-96:
    #   db_index = SeqIO.index(db_path, "fasta")
    #   db_keys = list(db_index.keys())
    #   record = db_index[db_keys[db_idx]]
    # This implies `db_keys` list maps integer index to ID. `db_index.keys()` order depends on implementation?
    # SeqIO.index usually preserves order? No, it's a dict.
    # But `list(db_index.keys())` might not be in file order if random access?
    # Actually `SeqIO.index` iterates file once to build index. It MIGHT preserve order in keys.
    # Let's trust `run_finetuned_full_eval.py` approach. It converts keys to list.
    
    # Re-fetch keys list to ensure mapping
    # Note: For very large DB, `list(db_index.keys())` consumes RAM.
    # `db_keys` variable logic in existing script is risky if order isn't guaranteed, but I will follow it.
    
    # Optimized aligner
    aligner = create_aligner("SW", 6, 1, "BLOSUM62")
    
    hits = []
    # Limit candidates to top 1000 like in fine-tuned script
    for db_idx in candidate_indices[:1000]:
        try:
            # We need to find the key corresponding to this index.
            # If we don't have db_keys pre-loaded, this is hard.
            # I will load db_keys list as done in reference script.
            if 'db_keys' not in locals():
                 db_keys = list(db_index.keys())
            
            record_id = db_keys[db_idx]
            record = db_index[record_id]
            
            q_s = re.sub(r'[^ARNDCQEGHILKMFPSTWYVBZX*]', 'X', query.sequence)
            r_s = re.sub(r'[^ARNDCQEGHILKMFPSTWYVBZX*]', 'X', str(record.seq))
            alns = aligner.align(q_s, r_s)
            
            if alns:
                aln = alns[0]
                start_q, end_q = aln.coordinates[0, 0], aln.coordinates[0, -1]
                
                # Check coverage > 20%
                if (end_q - start_q) >= len(query.sequence) * ALIGNMENT_COVERAGE_THRESHOLD:
                    seq_chars = ['X'] * start_q
                    for q_c, t_c in zip(aln[0], aln[1]):
                        if q_c != '-': 
                            seq_chars.append(t_c if t_c != '-' else '-')
                    
                    rem = len(query.sequence) - len(seq_chars)
                    if rem > 0: seq_chars.extend(['X'] * rem)
                    
                    final_seq = "".join(seq_chars)[:len(query.sequence)]
                    # Pad if short
                    if len(final_seq) < len(query.sequence): 
                        final_seq += 'X'*(len(query.sequence)-len(final_seq))
                    
                    hits.append(Chain(record_id, final_seq))
        except Exception as e:
            # print(f"Alignment error: {e}")
            pass

    print(f"Valid homologs aligned: {len(hits)}")

    final_score = None
    prob_deleterious = 0.0
    
    if len(hits) < MIN_HOMOLOGS:
        print(f"Insufficient homologs (< {MIN_HOMOLOGS}). Using fallback.")
        center_aa = sequence[mutation_pos]
        conservative = center_aa in "AGPST"
        final_score = 1.0 if conservative else 0.0
    else:
        # Calculate SIFT score
        sift_scores, num_used = sift_predictions_memory(hits, query, SEQ_IDENTITY_THRESHOLD, False)
        
        center_aa = sequence[mutation_pos]
        if center_aa in VALID_AAS:
            aa_idx = VALID_AAS.index(center_aa)
            score = sift_scores[mutation_pos][aa_idx]
            final_score = score
        else:
            print(f"Warning: Central AA '{center_aa}' not in standard 20 amino acids.")
            final_score = 0.0 # Treat as deleterious/unknown?
    
    if final_score is not None:
        prediction_label = "Tolerated" if final_score >= PREDICTION_CUTOFF else "Deleterious"
        prob_deleterious = 1.0 - final_score
        
        print("\n" + "="*40)
        print("PREDICTION RESULT")
        print("="*40)
        print(f"Sequence: {sequence}")
        print(f"SIFT Score: {final_score:.4f}")
        print(f"Probability Deleterious: {prob_deleterious:.4f}")
        print(f"Prediction: {prediction_label.upper()}")
        print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIFT 4G Single Sequence Prediction")
    parser.add_argument("sequence", nargs="?", help="Protein sequence (41-mer)")
    args = parser.parse_args()
    
    seq = args.sequence
    if not seq:
        print("Please enter the protein sequence:")
        seq = input().strip()
    
    predict(seq)
