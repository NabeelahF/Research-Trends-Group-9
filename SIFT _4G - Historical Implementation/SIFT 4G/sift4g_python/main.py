import argparse
import sys
import os
from Bio import SeqIO
from .utils import read_fasta_chains, create_file_name, query_log
from .database import search_database
from .alignment import create_aligner, align_sequences
from .prediction import sift_predictions

def main():
    parser = argparse.ArgumentParser(description="SIFT4G Python Implementation")
    
    parser.add_argument("-q", "--query", required=True, help="Input fasta query file")
    parser.add_argument("-d", "--database", required=True, help="Input fasta database file")
    parser.add_argument("--out", default=".", help="Output directory")
    parser.add_argument("--subst", default=".", help="Directory containing substitution files")
    
    # Parameters matches C++ defaults
    parser.add_argument("-k", "--kmer-length", type=int, default=5, help="Kmer length (3,4,5)")
    parser.add_argument("--max-candidates", type=int, default=5000, help="Max candidates per query")
    parser.add_argument("--max-aligns", type=int, default=400, help="Max alignments to keep")
    parser.add_argument("--eval-cutoff", type=float, default=0.0001, help="E-value cutoff (not fully implemented in python, using score rank)")
    
    # Alignment params
    parser.add_argument("--gap-open", type=int, default=10, help="Gap open penalty")
    parser.add_argument("--gap-extend", type=int, default=1, help="Gap extend penalty")
    parser.add_argument("--matrix", default="BLOSUM62", help="Score matrix")
    
    # Other
    parser.add_argument("--seq-identity", type=int, default=90, help="Sequence identity threshold") 
    parser.add_argument("--prob-cutoff", type=float, default=0.05, help="Deleterious prediction probability cutoff")
    
    args = parser.parse_args()
    
    if args.out and not os.path.exists(args.out):
        os.makedirs(args.out, exist_ok=True)
    
    # 1. Read Queries
    print(f"Reading queries from {args.query}...")
    queries = read_fasta_chains(args.query)
    if not queries:
        print("No queries found.")
        sys.exit(1)
        
    print(f"Read {len(queries)} queries.")
    
    # 2. Search Database
    print("Searching database for candidates...")
    # returns list of list of indices
    candidate_indices = search_database(queries, args.database, args.kmer_length, args.max_candidates)
    
    # 3. Retrieve Candidate Sequences and Align
    print("Aligning candidates...")
    
    # We need to retrieve sequences from DB by index. 
    # Bio.SeqIO.index is useful.
    try:
        db_index = SeqIO.index(args.database, "fasta")
        db_keys = list(db_index.keys()) # Map index 0..N to keys?
        # SeqIO.index uses header ID as key. 
        # But search_database used positional index 0..N.
        # We need mapping int -> key.
        # keys() returns iterator. Listing it might be memory heavy for huge DBs.
        # For huge DBs, C++ re-reads.
        # Here we will list keys (assuming reasonable size or memory)
        # If DB is huge, this is a bottleneck.
        pass
    except Exception as e:
        print(f"Error indexing database: {e}")
        sys.exit(1)

    aligner = create_aligner("SW", args.gap_open, args.gap_extend, args.matrix if args.matrix != "BLOSUM_62" else "BLOSUM62")
    
    for i, query in enumerate(queries):
        query_log(i+1, len(queries))
        
        indices = candidate_indices[i]
        if not indices:
            continue
            
        alignment_hits = []
        
        # Collect sequences
        # Optim: Sort indices to access linearly if iterating?
        # But we have random access via db_index (which is dict-like).
        
        for db_idx in indices:
            # Map int db_idx to key
            # This is slow if we do list(db_index.keys())[db_idx] every time.
            # Ideally search_database returned identifiers if feasible.
            # But search_database iterated generator.
            # We need a robust way.
            # Let's map db_idx -> key once.
            key = db_keys[db_idx]
            record = db_index[key]
            
            # Sanitize sequences
            # Define allowed alphabet for BLOSUM62 (BioPython's ExtendedIUPACProtein is broad, but Gapped/matrix might be strict)
            # Actually just replace U, O, J, B, Z, * with X or strict 20.
            # BLOSUM62 matrix usually has X, but we can't assume.
            # BioPython 1.78+ Align requires matching alphabet.
            
            def sanitize(seq_str):
                import re
                # Keep standard 20. Replace others with 'X'
                # But if 'X' is not in matrix support, might fail? 
                # BioPython substitution_matrices.load('BLOSUM62') supports:
                # A R N D C Q E G H I L K M F P S T W Y V B Z X *
                # So B, Z, X, * are allowed.
                # U, O, J are not.
                # Replace U, O, J with X.
                return re.sub(r'[^ARNDCQEGHILKMFPSTWYVBZX*]', 'X', seq_str)

            q_seq_sanitized = sanitize(query.sequence)
            r_seq_sanitized = sanitize(str(record.seq))

            # Align
            # Align query.sequence vs record.seq
            try:
                score = aligner.score(q_seq_sanitized, r_seq_sanitized)
            except ValueError:
                # If still fails, try stricter
                q_seq_sanitized = re.sub(r'[^ARNDCQEGHILKMFPSTWYV]', 'X', query.sequence)
                r_seq_sanitized = re.sub(r'[^ARNDCQEGHILKMFPSTWYV]', 'X', str(record.seq))
                score = aligner.score(q_seq_sanitized, r_seq_sanitized)

            # Create a simplified Chain object for the hit
            # We need the full alignment string/object for processing?
            # C++ stores alignment string (with gaps).
            
            # BioPython aligner returns alignment object.
            # Note: We align the sanitized versions, but mapped back? 
            # Actually alignment should be on sanitized.
            alignments = aligner.align(q_seq_sanitized, r_seq_sanitized)
            if alignments:
                aln = alignments[0]
                # We need the ALIGNED sequences (with gaps).
                # aln[0] is query aligned, aln[1] is target aligned.
                # Only need target aligned string?
                # C++ `sift_prediction` uses `alignment_string[currPos]`. 
                # Does it contain query or target?
                # In `siftAnalysis`, it usually aligns homologs (targets) to the query.
                # So we need the target sequence as it appears in the alignment (with gaps inserted to match query).
                # Wait, generic SIFT inputs aligned sequences (MSA).
                # SIFT4G does the alignment.
                # So we need the aligned version of the DB sequence.
                
                # BioPython: aln[0] is aligned query, aln[1] is aligned target.
                # But they might have gaps relative to each other.
                # SIFT expects all sequences derived from an MSA, implicitly aligned to the query.
                # So we need the target sequence with gaps, such that it matches the query length?
                # If local alignment, they might be partial.
                # SIFT requires global-ish view or mapping to query positions.
                # C++ uses local alignment (SW).
                
                # Warning: Python implementation of MSA from SW pairwise is tricky.
                # We will essentially take the aligned part of the target.
                # And pad?
                # Just using valid chars.
                
                # Let's trust that aln[1] (target) with gaps is what we want, 
                # but we must ensure it maps to query indices 0..N?
                # If local alignment starts at query pos 10, positions 0..9 are missing.
                # We should pad with 'X' or '-'?
                # C++ `sift_prediction.cpp`: `remove_seqs_percent_identical_to_query` loops `m` from 0 to `lenOfAlign`.
                # And check `lenOfQuery == lenOfAlign`.
                # So the alignment string MUST be same length as query.
                
                # So we need to construct a full length string for the target, matching query length.
                # 1. Fill with '-' or 'X' for non-aligned regions.
                # 2. Place aligned residues.
                
                # BioPython alignment coordinates:
                # aln.indices gives indices of residues.
                
                aligned_target_full = ['-'] * len(query.sequence)
                
                # aln.coordinates? 
                # A generic way: iterate the alignment.
                # aln[0] is query with gaps.
                # aln[1] is target with gaps.
                # We iterate them together.
                # We need to map to ORIGINAL query positions.
                
                q_aln = aln[0]
                t_aln = aln[1]
                
                q_pos = 0 # Index in original query

                
                # Robust alignment string construction
                try:
                    # Get alignment strings
                    # BioPython 1.80+ PairwiseAligner alignment object is roughly:
                    # alignment[0] -> aligned query sequence string
                    # alignment[1] -> aligned target sequence string
                    # coordinates -> [ [start_row, start_col], ... ]
                    
                    # For local alignment, we assume coordinates start is where match begins.
                    # We need to PAD the target string to match Query length.
                    
                    # 1. Get query start offset
                    # alignment.coordinates is shape (2, N_segments+1)?
                    # Or alignment.path?
                    # Simpler: alignment[0] string vs query string check?
                    # Actually standard way:
                    
                    aligned_q = aln[0]
                    aligned_t = aln[1]
                    
                    # We need to calculate how many query residues are BEFORE this alignment starts
                    # aln.coordinates[0, 0] gives the start index in sequence 0 (query)
                    
                    start_q = aln.coordinates[0, 0]
                    end_q = aln.coordinates[0, -1] # Last index
                    
                    # Construct valid target string
                    # Fill 'X' for unaligned start
                    constructed_seq = ['X'] * start_q
                    
                    # Iterate aligned region
                    # If aligned_q has char -> it corresponds to a query position -> take aligned_t char
                    # If aligned_q has gap -> insertion in target -> skip
                    
                    for q_char, t_char in zip(aligned_q, aligned_t):
                        if q_char != '-':
                            # Corresponds to a query position
                            # t_char might be residue or gap ('-')
                            constructed_seq.append(t_char if t_char != '-' else '-')
                    
                    # Fill 'X' for unaligned end
                    remaining = len(query.sequence) - len(constructed_seq)
                    if remaining > 0:
                        constructed_seq.extend(['X'] * remaining)
                        
                    # Final string
                    final_seq_str = "".join(constructed_seq)
                    
                    # Verify length
                    if len(final_seq_str) != len(query.sequence):
                        # Should not happen with correct logic
                        # Truncate or pad if edge case
                        if len(final_seq_str) > len(query.sequence):
                            final_seq_str = final_seq_str[:len(query.sequence)]
                        else:
                            final_seq_str += 'X' * (len(query.sequence) - len(final_seq_str))
                    
                    # Store as Chain
                    # Use db_idx or key as header
                    # Header must enable looking up substitution? Not strictly used in Prediction except for name?
                    # main.cpp: output uses query Name.
                    # The alignment sequences names are used in `printSeqNames`?
                    # Name is just identifier.
                    
                    from .utils import Chain
                    hit_chain = Chain(key, final_seq_str)
                    alignment_hits.append(hit_chain)
                    
                except Exception as e:
                    # Fallback or log
                    # print(f"Alignment processing error: {e}")
                    pass
        
        # 4. Filter and Predict
        if alignment_hits:
            sift_predictions(alignment_hits, query, args.subst, args.seq_identity, args.out, args.prob_cutoff)

    print("Predictions complete.")

if __name__ == "__main__":
    main()
