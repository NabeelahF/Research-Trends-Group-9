"""
In-memory SIFT4G evaluation without saving prediction files.
This script runs predictions and evaluations directly in memory to avoid disk I/O.
"""
import argparse
import csv
from Bio import SeqIO
from sklearn.metrics import classification_report, roc_auc_score

# Import SIFT modules using relative imports
from .utils import read_fasta_chains, Chain
from .database import search_database
from .alignment import create_aligner
from .prediction import (
    remove_seqs_percent_identical_to_query,
    calc_sift_scores,
    create_matrix
)
import re


def sanitize_sequence(seq_str):
    """Sanitize sequence for alignment"""
    return re.sub(r'[^ARNDCQEGHILKMFPSTWYVBZX*]', 'X', seq_str)


def run_sift_prediction_inmemory(query, alignment_hits, seq_identity=90):
    """
    Run SIFT prediction in memory without writing files.
    Returns the SIFT score matrix.
    """
    kMaxSequences = 400
    
    # Limit sequences
    if len(alignment_hits) > kMaxSequences - 1:
        alignment_hits = alignment_hits[:kMaxSequences - 1]
    
    query_length = len(query.sequence)
    
    # Remove highly similar sequences
    remove_seqs_percent_identical_to_query(query, alignment_hits, seq_identity)
    
    # Add query to beginning
    alignment_hits.insert(0, query)
    
    # Initialize matrices
    matrix = [[0.0]*26 for _ in range(query_length)]
    sift_scores = [[0.0]*26 for _ in range(query_length)]
    
    # Calculate initial matrix
    weights_1_temp = [1.0] * len(alignment_hits)
    aas_stored_temp = [0.0] * query_length
    create_matrix(alignment_hits, query, weights_1_temp, matrix, aas_stored_temp)
    
    # Calculate SIFT scores
    calc_sift_scores(alignment_hits, query, matrix, sift_scores)
    
    return sift_scores


def get_sift_score_for_variant(sift_matrix, position, alt_aa):
    """Extract SIFT score for a specific variant"""
    if position < 1 or position > len(sift_matrix):
        return None
    
    aa_index = ord(alt_aa) - ord('A')
    if aa_index < 0 or aa_index >= 26:
        return None
    
    return sift_matrix[position - 1][aa_index]


def main():
    parser = argparse.ArgumentParser(description="SIFT4G In-Memory Evaluation")
    parser.add_argument("-q", "--query", required=True, help="Query FASTA file")
    parser.add_argument("-d", "--database", required=True, help="Database FASTA file")
    parser.add_argument("--truth", required=True, help="Ground truth CSV file")
    parser.add_argument("--kmer-length", type=int, default=5, help="K-mer length")
    parser.add_argument("--max-candidates", type=int, default=5000, help="Max candidates")
    parser.add_argument("--max-aligns", type=int, default=400, help="Max alignments")
    parser.add_argument("--gap-open", type=int, default=10, help="Gap open penalty")
    parser.add_argument("--gap-extend", type=int, default=1, help="Gap extend penalty")
    parser.add_argument("--matrix", default="BLOSUM62", help="Scoring matrix")
    parser.add_argument("--seq-identity", type=int, default=90, help="Sequence identity threshold")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SIFT4G In-Memory Evaluation (No File Writing)")
    print("=" * 60)
    
    # Load ground truth
    print(f"\nLoading ground truth from {args.truth}...")
    ground_truth = {}
    with open(args.truth, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            protein = row['Protein']
            if protein not in ground_truth:
                ground_truth[protein] = []
            ground_truth[protein].append({
                'position': int(row['Position']),
                'alt': row['Alt'],
                'label': row['Label'].strip().lower()
            })
    
    print(f"Loaded {len(ground_truth)} proteins with variants")
    
    # Load queries
    print(f"\nReading queries from {args.query}...")
    queries = read_fasta_chains(args.query)
    print(f"Loaded {len(queries)} query sequences")
    
    # Index database
    print(f"\nIndexing database {args.database}...")
    db_index = SeqIO.index(args.database, "fasta")
    db_keys = list(db_index.keys())
    print(f"Database indexed with {len(db_keys)} sequences")
    
    # Search database for all queries
    print(f"\nSearching database for homologs...")
    candidate_indices = search_database(queries, args.database, args.kmer_length, args.max_candidates)
    
    # Create aligner
    aligner = create_aligner("SW", args.gap_open, args.gap_extend, args.matrix)
    
    # Evaluation metrics
    y_true = []
    y_pred = []
    y_scores = []
    
    tp = tn = fp = fn = 0
    processed_proteins = 0
    skipped_proteins = 0
    
    print(f"\nProcessing predictions and evaluating...")
    
    for i, query in enumerate(queries):
        # Extract protein ID from query header
        # Format: sp|P12345|NAME or just P12345
        header_parts = query.header.split('|')
        if len(header_parts) >= 2:
            protein_id = header_parts[1]
        else:
            protein_id = query.header.split()[0]
        
        # Check if this protein has ground truth
        if protein_id not in ground_truth:
            skipped_proteins += 1
            continue
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(queries)} queries...")
        
        # Get candidate indices
        indices = candidate_indices[i]
        if not indices:
            skipped_proteins += 1
            continue
        
        # Align candidates
        alignment_hits = []
        for db_idx in indices[:args.max_aligns]:
            key = db_keys[db_idx]
            record = db_index[key]
            
            q_seq = sanitize_sequence(query.sequence)
            r_seq = sanitize_sequence(str(record.seq))
            
            try:
                score = aligner.score(q_seq, r_seq)
                alignments = aligner.align(q_seq, r_seq)
                
                if alignments:
                    aln = alignments[0]
                    aligned_q = aln[0]
                    aligned_t = aln[1]
                    
                    start_q = aln.coordinates[0, 0]
                    
                    # Construct aligned target sequence
                    constructed_seq = ['X'] * start_q
                    for q_char, t_char in zip(aligned_q, aligned_t):
                        if q_char != '-':
                            constructed_seq.append(t_char if t_char != '-' else '-')
                    
                    remaining = len(query.sequence) - len(constructed_seq)
                    if remaining > 0:
                        constructed_seq.extend(['X'] * remaining)
                    
                    final_seq_str = "".join(constructed_seq)
                    
                    if len(final_seq_str) > len(query.sequence):
                        final_seq_str = final_seq_str[:len(query.sequence)]
                    elif len(final_seq_str) < len(query.sequence):
                        final_seq_str += 'X' * (len(query.sequence) - len(final_seq_str))
                    
                    hit_chain = Chain(key, final_seq_str)
                    alignment_hits.append(hit_chain)
            except Exception:
                continue
        
        if not alignment_hits:
            skipped_proteins += 1
            continue
        
        # Run SIFT prediction in memory
        try:
            sift_matrix = run_sift_prediction_inmemory(query, alignment_hits, args.seq_identity)
        except Exception as e:
            print(f"  Warning: SIFT prediction failed for {protein_id}: {e}")
            skipped_proteins += 1
            continue
        
        # Evaluate against ground truth
        for variant in ground_truth[protein_id]:
            pos = variant['position']
            alt = variant['alt']
            label = variant['label']
            
            score = get_sift_score_for_variant(sift_matrix, pos, alt)
            if score is None:
                continue
            
            is_pathogenic = label in ['deleterious', 'pathogenic', 'positive', 'disease']
            predicted_deleterious = score < 0.05
            
            # Store for metrics
            y_true.append(1 if is_pathogenic else 0)
            y_pred.append(1 if predicted_deleterious else 0)
            y_scores.append(1 - score)  # Invert for AUC
            
            # Update confusion matrix
            if is_pathogenic:
                if predicted_deleterious:
                    tp += 1
                else:
                    fn += 1
            else:
                if predicted_deleterious:
                    fp += 1
                else:
                    tn += 1
        
        processed_proteins += 1
    
    print(f"\nProcessed {processed_proteins} proteins, skipped {skipped_proteins}")
    
    # Calculate metrics
    total = tp + tn + fp + fn
    if total == 0:
        print("\nNo valid comparisons found.")
        return
    
    accuracy = (tp + tn) / total
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    try:
        auc = roc_auc_score(y_true, y_scores)
    except Exception as e:
        auc = None
        print(f"Warning: Could not calculate AUC: {e}")
    
    # Print results
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS ({total} mutations)")
    print("=" * 60)
    
    if auc is not None:
        print(f"\nAUC (Area Under ROC Curve): {auc:.4f}")
    
    print(f"\nAccuracy: {accuracy:.2%}")
    print(f"  - Sensitivity (Recall): {sensitivity:.2%}")
    print(f"  - Specificity: {specificity:.2%}")
    print(f"  - Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    
    print("\n" + "-" * 60)
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Tolerated', 'Deleterious']))


if __name__ == "__main__":
    main()
