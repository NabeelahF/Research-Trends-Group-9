#!/usr/bin/env python3
"""
SIFT 4G Baseline Predictions (No Fine-tuning)
Uses default SIFT parameters on processed_dataset (1).csv
"""

import os
import sys
import csv
import re
import copy
from Bio import SeqIO
from collections import defaultdict
import numpy as np

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
print_matrix_original_format = prediction.print_matrix_original_format
TOLERANCE_PROB_THRESHOLD = prediction.TOLERANCE_PROB_THRESHOLD

KMER_LENGTH = 5
DEFAULT_MAX_CANDIDATES = 20000
kMaxSequences = 800

def parse_sift_score(filepath, center_pos=20):
    """Parse SIFT score from prediction file at center position."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        
        data_lines = []
        for line in lines:
            parts = line.split()
            if parts and parts[0].replace('.','',1).replace('-','',1).isdigit():
                data_lines.append(parts)
        
        if center_pos < len(data_lines):
            scores = [float(x) for x in data_lines[center_pos][1:]]
            return scores
    except:
        pass
    return None

def sift_predictions_custom(alignment_strings, query, subst_path, sequence_identity, out_path, prediction_cutoff=0.05, exclude_query=False):
    """Run SIFT prediction with given parameters."""
    if len(alignment_strings) > kMaxSequences - 1:
        alignment_strings[:] = alignment_strings[:kMaxSequences - 1]
        
    query_length = len(query.sequence)
    
    # Standard filtering
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
    
    out_extension = ".SIFTprediction"
    out_file_name = create_file_name(query.header.split()[0], out_path, out_extension)
    
    print_matrix_original_format(sift_scores, out_file_name)
    
    return len(alignment_strings)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    query_path = os.path.join(base_dir, "processed_dataset.fasta")
    db_path = os.path.join(base_dir, "uniprot_sprot.fasta")
    gt_path = os.path.join(base_dir, "processed_dataset (1).csv")
    
    print("="*80)
    print("SIFT 4G - BASELINE PREDICTIONS (NO FINE-TUNING)")
    print("="*80)
    print("Using default SIFT parameters:")
    print("  - Sequence Identity: 90%")
    print("  - Prediction Cutoff: 0.05")
    print("  - Minimum Homologs: 10")
    print("="*80)
    
    print("\nLoading ground truth...")
    gt_map = {}
    with open(gt_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = row['sequence_window'].strip()
            label = int(row['label'])
            gt_map[seq] = label
    
    print(f"Loaded {len(gt_map)} labeled sequences")
    
    all_queries = read_fasta_chains(query_path)
    labeled_queries = [q for q in all_queries if q.sequence in gt_map][:2000]
    print(f"Using {len(labeled_queries)} query sequences")
    
    print(f"\nSearching database...")
    candidate_indices = search_database(labeled_queries, db_path, KMER_LENGTH, DEFAULT_MAX_CANDIDATES)
    
    print("Aligning candidates...")
    db_index = SeqIO.index(db_path, "fasta")
    db_keys = list(db_index.keys())
    
    # Standard alignment parameters
    aligner = create_aligner("SW", 10, 1, "BLOSUM62")
    
    all_alignment_hits = [] 
    homolog_stats = []
    
    for i, query in enumerate(labeled_queries):
        if i % 50 == 0: 
            print(f"Processing {i+1}/{len(labeled_queries)}...")
        hits = []
        indices = candidate_indices[i]
        
        if indices:
            for db_idx in indices[:500]:  # Standard candidate limit
                try:
                    record = db_index[db_keys[db_idx]]
                    q_s = re.sub(r'[^ARNDCQEGHILKMFPSTWYVBZX*]', 'X', query.sequence)
                    r_s = re.sub(r'[^ARNDCQEGHILKMFPSTWYVBZX*]', 'X', str(record.seq))
                    alns = aligner.align(q_s, r_s)
                    
                    if alns:
                        aln = alns[0]
                        start_q = aln.coordinates[0, 0]
                        end_q = aln.coordinates[0, -1]
                        aligned_length = end_q - start_q
                        
                        # Standard coverage requirement: 50%
                        if aligned_length >= len(query.sequence) * 0.5:
                            seq_chars = ['X'] * start_q
                            for q_c, t_c in zip(aln[0], aln[1]):
                                if q_c != '-': 
                                    seq_chars.append(t_c if t_c != '-' else '-')
                            rem = len(query.sequence) - len(seq_chars)
                            if rem > 0: 
                                seq_chars.extend(['X'] * rem)
                            final_seq = "".join(seq_chars)[:len(query.sequence)]
                            if len(final_seq) < len(query.sequence): 
                                final_seq += 'X'*(len(query.sequence)-len(final_seq))
                            hits.append(Chain(db_keys[db_idx], final_seq))
                except: 
                    pass
        
        all_alignment_hits.append(hits)
        homolog_stats.append(len(hits))
    
    # Diagnostic output
    print("\n" + "="*60)
    print("HOMOLOG STATISTICS")
    print("="*60)
    print(f"Mean homologs per query: {np.mean(homolog_stats):.1f}")
    print(f"Median homologs: {np.median(homolog_stats):.1f}")
    print(f"Min homologs: {np.min(homolog_stats)}")
    print(f"Max homologs: {np.max(homolog_stats)}")
    print(f"Queries with <10 homologs: {sum(1 for h in homolog_stats if h < 10)}/{len(homolog_stats)}")
    print(f"Queries with <50 homologs: {sum(1 for h in homolog_stats if h < 50)}/{len(homolog_stats)}")

    print("\nRunning SIFT predictions...")
    out_dir = os.path.join(base_dir, "results_baseline_temp")
    if not os.path.exists(out_dir): 
        os.makedirs(out_dir)
    
    # BASELINE configuration (standard SIFT parameters)
    seq_identity = 90  # Standard
    cutoff = 0.05  # Standard SIFT cutoff
    exclude_query = False
    min_homologs = 10  # Standard minimum
    
    tp = tn = fp = fn = 0
    results_list = []
    valid_aas = "ACDEFGHIKLMNPQRSTVWY"
    skipped = 0
    
    score_distribution = {"deleterious": [], "tolerated": []}
    
    for i, query in enumerate(labeled_queries):
        if i % 50 == 0:
            print(f"Predicting {i+1}/{len(labeled_queries)}...")
            
        current_hits = copy.deepcopy(all_alignment_hits[i])
        
        if len(current_hits) < min_homologs:
            skipped += 1
            final_score = "N/A"
            prediction = "N/A"
            status = "Skipped"
            num_homologs = len(current_hits)
        else:
            final_aligned = sift_predictions_custom(current_hits, query, ".", seq_identity, out_dir, cutoff, exclude_query)
            
            pred_file = os.path.join(out_dir, f"{query.header.split()[0]}.SIFTprediction")
            scores = parse_sift_score(pred_file)
            
            final_score = "N/A"
            status = "Unknown"
            prediction = "N/A"
            num_homologs = final_aligned
            
            if scores is not None:
                try:
                    center_aa = query.sequence[20]
                    
                    if center_aa in valid_aas:
                        aa_idx = valid_aas.index(center_aa)
                        score = scores[aa_idx]
                        final_score = f"{score:.4f}"
                        
                        # Track score distribution
                        lbl = gt_map[query.sequence]
                        if lbl == 1:
                            score_distribution["deleterious"].append(score)
                        else:
                            score_distribution["tolerated"].append(score)
                        
                        is_del_pred = (score < cutoff)
                        is_del_lbl = (lbl == 1)
                        
                        prediction = "Deleterious" if is_del_pred else "Tolerated"
                        
                        if is_del_lbl:
                            if is_del_pred: 
                                tp += 1
                                status = "TP"
                            else: 
                                fn += 1
                                status = "FN"
                        else:
                            if is_del_pred: 
                                fp += 1
                                status = "FP"
                            else: 
                                tn += 1
                                status = "TN"
                except:
                    pass
        
        lbl = gt_map[query.sequence]
        results_list.append({
            "SequenceID": query.header.split()[0],
            "Sequence": query.sequence,
            "CenterAA": query.sequence[20],
            "Num_Homologs": num_homologs,
            "SIFT_Score": final_score,
            "Prediction": prediction,
            "True_Label": "Deleterious" if lbl == 1 else "Tolerated",
            "Status": status
        })

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print("\n" + "="*60)
    print("BASELINE RESULTS")
    print("="*60)
    print(f"Parameters: ID={seq_identity}%, Cutoff={cutoff}, MinHom={min_homologs}")
    print(f"Total Sequences: {total}")
    print(f"Skipped (low homologs): {skipped}")
    print(f"Accuracy: {acc:.2%}")
    print(f"Sensitivity: {sens:.2%}")
    print(f"Specificity: {spec:.2%}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    
    # Score distribution analysis
    if score_distribution["deleterious"] and score_distribution["tolerated"]:
        print("\n" + "="*60)
        print("SCORE DISTRIBUTION ANALYSIS")
        print("="*60)
        print(f"Deleterious variants - Mean score: {np.mean(score_distribution['deleterious']):.4f}")
        print(f"Tolerated variants - Mean score: {np.mean(score_distribution['tolerated']):.4f}")
    
    csv_path = os.path.join(base_dir, "predictions_baseline.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["SequenceID", "Sequence", "CenterAA", "Num_Homologs", "SIFT_Score", "Prediction", "True_Label", "Status"])
        writer.writeheader()
        writer.writerows(results_list)
    
    print(f"\nPredictions saved to: {csv_path}")
    
    import shutil
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        print("Temporary files cleaned up")

if __name__ == "__main__":
    main()
