import bisect
from collections import defaultdict
import sys
import utils
database_log = utils.database_log

def get_kmers(sequence, k):
    """
    Generates k-mers from a sequence.
    Returns a list of k-mers.
    """
    n = len(sequence)
    if n < k:
        return []
    return [sequence[i:i+k] for i in range(n - k + 1)]

def create_query_index(queries, k):
    """
    Creates an index of k-mers from query sequences.
    Returns a dictionary mapping k-mer to list of (query_index, position).
    """
    index = defaultdict(list)
    for q_idx, query in enumerate(queries):
        seq = query.sequence
        for kmer_str in get_kmers(seq, k):
            # We record position, but in C++ it records position for LIS.
            # wait, C++: hits[begin->id].emplace_back(begin->position);
            # Yes, position in query? No, position in DB or Query?
            # createHash(queries...) -> Hash of QUERIES.
            # threadSearchDatabase: createKmerVector(db_seq).
            # match db_kmer to query_hash.
            # begin->position comes from query hash.
            # So it's position in QUERY.
            # Actually, `params` to createHash includes `val_limit`?
            # Creating k-mers for queries.
            # So index stores (query_id, query_pos).
            # get_kmers returns strings.
            pass
        
        # Optimization: use explicit loop to get pos
        seq = query.sequence
        n = len(seq)
        if n >= k:
            for i in range(n - k + 1):
                kmer = seq[i:i+k]
                index[kmer].append((q_idx, i))
    return index

def longest_increasing_subsequence(sequence):
    """
    Computes the length of the Longest Increasing Subsequence (LIS).
    O(n log n)
    """
    if not sequence:
        return 0
    
    # helper array to store smallest tail of all increasing subsequences of length i+1
    tails = []
    
    for x in sequence:
        idx = bisect.bisect_left(tails, x)
        if idx < len(tails):
            tails[idx] = x
        else:
            tails.append(x)
            
    return len(tails)

class Candidate:
    def __init__(self, score, db_idx):
        self.score = score
        self.db_idx = db_idx
    
    def __lt__(self, other):
        # We want higher score to be "better", but for sorting to keep top N...
        # Python min-heap keeps smallest.
        # We want to keep LARGEST scores.
        # So if we sort descending, or use heap.
        return self.score < other.score

def search_database(queries, database_path, kmer_length=5, max_candidates=5000):
    """
    Searches the database for sequences similar to queries.
    Returns a list of lists of candidate indices (one list per query).
    """
    import utils # Lazy import to avoid cycle if any
    read_fasta_chains = utils.read_fasta_chains
    
    # 1. Index Queries
    query_index = create_query_index(queries, kmer_length)
    
    candidates = [[] for _ in range(len(queries))] # List of lists of Candidate objects
    
    # 2. Iterate Database
    # We need to read the database. For large DBs, this should be chunked.
    # Using our utils.read_fasta_chains implies reading whole into memory.
    # C++ reads in chunks.
    # Python Bio.SeqIO.parse is an iterator.
    
    from Bio import SeqIO
    if not database_path:
        return []

    db_idx = 0
    try:
        # Use iterator to avoid loading full DB
        iterator = SeqIO.parse(database_path, "fasta")
        
        # For logging
        # We don't know total size easily without reading.
        # C++ estimates or tracks bytes.
        
        for record in iterator:
            seq_str = str(record.seq)
            db_len = len(seq_str)
            
            # Find hits
            # hits[query_id] = [query_pos, ...]
            hits = defaultdict(list)
            
            kmers = get_kmers(seq_str, kmer_length)
            
            # Optimization: filter unique congruent kmers? C++: if (j != 0 && kmer_vector[j] == kmer_vector[j - 1]) continue;
            # C++ creates vector then sorts it? No, `createKmerVector` does something.
            # In Python, just iterating.
            
            for kmer in kmers:
                if kmer in query_index:
                    for q_idx, q_pos in query_index[kmer]:
                        hits[q_idx].append(q_pos)
            
            # Score
            for q_idx, q_hits in hits.items():
                if not q_hits:
                    continue
                
                # q_hits contains positions in query.
                # We want LIS of these positions.
                # Since we iterate DB seq linearly, the "time" axis is DB position.
                # The "value" axis is Query position.
                # If DB and Query match, the Query positions should be increasing as we move through DB.
                # So just LIS of q_hits (which are appended in order of DB occurrence).
                
                q_len = len(queries[q_idx].sequence)
                denom = min(float(db_len), float(q_len))
                score_val = longest_increasing_subsequence(q_hits) / denom if denom > 0 else 0
                
                # Add to candidates
                # Maintaining top N
                # If list < max, add.
                # If list full and score > min, replace min.
                
                c_list = candidates[q_idx]
                if len(c_list) < max_candidates:
                    c_list.append(Candidate(score_val, db_idx))
                    # Retaining sorted order not strictly needed every step, but helpful for "min".
                    # We can sort later or use heap.
                    # Simple: Just append and sort/prune occasionally or at end?
                    # C++ prunes per thread.
                    # Implementing simple prune check at end of DB or periodically?
                    # For performance in Python, let's prune only at end/chunks?
                    # But memory might be issue.
                    # Let's prune if len > 2 * max_candidates?
                else:
                     # Simple logic: append, sort descending, slice.
                     c_list.append(Candidate(score_val, db_idx))
            
            db_idx += 1
            if db_idx % 1000 == 0:
                database_log(1, 0, 0) # Dummy progress
                
                # Prune periodically
                for q_i in range(len(queries)):
                     if len(candidates[q_i]) > 2 * max_candidates:
                         candidates[q_i].sort(key=lambda x: x.score, reverse=True)
                         candidates[q_i] = candidates[q_i][:max_candidates]

    except Exception as e:
        sys.stderr.write(f"Error reading database: {e}\n")
        return []

    # Final Prune and format
    final_indices = []
    for c_list in candidates:
        c_list.sort(key=lambda x: x.score, reverse=True)
        top = c_list[:max_candidates]
        # Return only indices, sorted by index?
        # C++: returns indices sorted.
        indices = sorted([c.db_idx for c in top])
        final_indices.append(indices)
        
    return final_indices
