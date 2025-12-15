#!/usr/bin/env python3
"""
SIFT 4G Baseline Predictions with Full Evaluation (In-Memory)
Runs standard SIFT (No fine-tuning) on 2000 queries.
Optimized to avoid file I/O.
"""

import os
import sys
import csv
import re
import copy
from Bio import SeqIO
from collections import defaultdict
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

import utils
import database
import alignment
import prediction

read_fasta_chains = utils.read_fasta_chains
create_file_name = utils.create_file_name
Chain = utils.Chain
query_log = utils.query_log
search_database = database.search_database
create_aligner = alignment.create_aligner
remove_seqs_percent_identical_to_query = prediction.remove_seqs_percent_identical_to_query
create_matrix = prediction.create_matrix
calc_sift_scores = prediction.calc_sift_scores

KMER_LENGTH = 5
DEFAULT_MAX_CANDIDATES = 20000
kMaxSequences = 800

def sift_predictions_memory(alignment_strings, query, sequence_identity, exclude_query=False):
    """
    Run SIFT prediction and return scores directly (No file I/O).
    Returns (sift_scores_matrix, num_sequences_used)
    """
    if len(alignment_strings) > kMaxSequences - 1:
        alignment_strings[:] = alignment_strings[:kMaxSequences - 1]
    
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

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    query_path = os.path.join(base_dir, "processed_dataset.fasta")
    db_path = os.path.join(base_dir, "uniprot_sprot.fasta")
    gt_path = os.path.join(base_dir, "processed_dataset (1).csv")
    
    print("="*80)
    print("SIFT 4G - BASELINE EVALUATION (NO FINE-TUNING)")
    print("="*80)
    print("Using Standard SIFT Parameters:")
    print("  - Sequence Identity: 90%")
    print("  - Prediction Cutoff: 0.05")
    print("  - Minimum Homologs: 10")
    print("  - Alignment Coverage: 50%")
    print("  - Target Sequences: 2000")
    print("="*80)
    
    # Load labels
    gt_map = {}
    with open(gt_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_map[row['sequence_window'].strip()] = int(row['label'])
            
    all_queries = read_fasta_chains(query_path)
    labeled_queries = [q for q in all_queries if q.sequence in gt_map][:2000] # TARGET 2000
    print(f"Processing {len(labeled_queries)} sequences")
    
    print("Searching database...")
    candidate_indices = search_database(labeled_queries, db_path, KMER_LENGTH, DEFAULT_MAX_CANDIDATES)
    
    print("Aligning candidates (Standard SW parameters)...")
    db_index = SeqIO.index(db_path, "fasta")
    db_keys = list(db_index.keys())
    
    # BASELINE ALIGNMENT PARAMETERS (Standard SIFT)
    aligner = create_aligner("SW", 10, 1, "BLOSUM62")
    
    all_alignment_hits = [] 
    for i, query in enumerate(labeled_queries):
        hits = []
        if candidate_indices[i]:
            for db_idx in candidate_indices[i][:2000]: # Standard limits
                try:
                    record = db_index[db_keys[db_idx]]
                    q_s = re.sub(r'[^ARNDCQEGHILKMFPSTWYVBZX*]', 'X', query.sequence)
                    r_s = re.sub(r'[^ARNDCQEGHILKMFPSTWYVBZX*]', 'X', str(record.seq))
                    alns = aligner.align(q_s, r_s)
                    if alns:
                        aln = alns[0]
                        start_q, end_q = aln.coordinates[0, 0], aln.coordinates[0, -1]
                        if (end_q - start_q) >= len(query.sequence) * 0.5: # Standard 50% coverage
                            seq_chars = ['X'] * start_q
                            for q_c, t_c in zip(aln[0], aln[1]):
                                if q_c != '-': seq_chars.append(t_c if t_c != '-' else '-')
                            rem = len(query.sequence) - len(seq_chars)
                            if rem > 0: seq_chars.extend(['X'] * rem)
                            final_seq = "".join(seq_chars)[:len(query.sequence)]
                            if len(final_seq) < len(query.sequence): final_seq += 'X'*(len(query.sequence)-len(final_seq))
                            hits.append(Chain(db_keys[db_idx], final_seq))
                except: pass
        all_alignment_hits.append(hits)
        if i % 100 == 0: print(f"Aligned {i+1}...")

    print("\nRunning Predictions...")
    
    valid_aas = "ACDEFGHIKLMNPQRSTVWY"
    
    y_true = []
    y_scores = [] 
    y_pred = []
    
    min_homologs = 10 # Standard SIFT min
    cutoff = 0.05 # Standard SIFT cutoff
    
    skipped_count = 0
    
    for i, query in enumerate(labeled_queries):
        if i % 200 == 0: print(f"Predicting {i+1}...")
        
        hits = copy.deepcopy(all_alignment_hits[i])
        
        final_score = None
        
        if len(hits) < min_homologs:
            # Baseline behavior: Skip or treat as low confidence.
            # Usually SIFT won't predict. We'll mark as N/A or default to Tolerated.
            # For consistent evaluation, we skip these from metrics calculation 
            # OR default them. SIFT usually doesn't output.
            skipped_count += 1
        else:
            # IN-MEMORY CALCULATION
            sift_scores, num_used = sift_predictions_memory(hits, query, 90, False) # 90% ID filter
            
            center_aa = query.sequence[20]
            if center_aa in valid_aas:
                aa_idx = valid_aas.index(center_aa)
                score = sift_scores[20][aa_idx]
                final_score = score
        
        lbl = gt_map[query.sequence]
        
        if final_score is not None:
            # We have a valid prediction
            y_true.append(lbl)
            y_scores.append(1.0 - final_score) # Prob of Deleterious
            y_pred.append(1 if final_score < cutoff else 0)
            
    print("\n" + "="*80)
    print("EVALUATION REPORT")
    print("="*80)
    
    if len(y_true) == 0:
        print("No valid predictions made (insufficient homologs?)")
        return

    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_scores)
    except Exception as e:
        auc = f"N/A ({e})"

    print(f"Total Evaluated Sequences: {len(y_true)}")
    print(f"Skipped Sequences (<{min_homologs} homologs): {skipped_count}")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC Score: {auc if isinstance(auc, str) else f'{auc:.4f}'}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Tolerated", "Deleterious"]))
    
    print("="*80)

if __name__ == "__main__":
    main()
