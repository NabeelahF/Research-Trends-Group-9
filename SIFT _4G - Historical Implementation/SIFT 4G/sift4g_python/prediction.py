import math
import sys
import re
import constants
import utils

rank_matrix = constants.rank_matrix
diri_q = constants.diri_q
diri_alpha = constants.diri_alpha
diri_altot = constants.diri_altot
aa_frequency = constants.aa_frequency
kLog_2_20 = constants.kLog_2_20
create_file_name = utils.create_file_name

# Constants
kMaxSequences = 400
TOLERANCE_PROB_THRESHOLD = 0.05
ADEQUATE_SEQ_INFO = 3.25

def valid_amino_acid(aa):
    if aa in ['B', 'Z', 'J', 'O', 'U', 'X', '-', '*']:
        return False
    return True

def aa_to_idx(aa):
    return ord(aa) - ord('A')

def add_logs(logx, logy):
    if logx > logy:
        return logx + math.log(1.0 + math.exp(logy - logx))
    else:
        return logy + math.log(1.0 + math.exp(logx - logy))

def remove_seqs_percent_identical_to_query(query, alignment_strings, seq_identity):
    """
    Removes sequences from alignment_strings that are >= seq_identity identical to query.
    alignment_strings is a list of Chain objects. Modified in-place.
    """
    len_of_query = len(query.sequence)
    to_remove = []
    
    for i, seq_chain in enumerate(alignment_strings):
        identity = 0
        seq_total = 0
        len_of_align = len(seq_chain.sequence)
        
        if len_of_query != len_of_align:
            # Should error or skip? C++ exits.
            # We skip for now or warn.
            continue
            
        for m in range(len_of_align):
            q_char = query.sequence[m]
            a_char = seq_chain.sequence[m]
            
            if a_char != 'X':
                if valid_amino_acid(a_char) and valid_amino_acid(q_char):
                    seq_total += 1
                    if q_char == a_char:
                        identity += 1
        
        if seq_total > 0:
            perc_similar = (identity / seq_total) * 100
        else:
            perc_similar = 0
            
        if perc_similar >= seq_identity:
            to_remove.append(i)
            
    # Remove in reverse order to maintain indices
    for i in sorted(to_remove, reverse=True):
        del alignment_strings[i]

def calc_seq_weights(alignment_strings, matrix, seq_weights, number_of_diff_aas):
    """
    Calculates sequence weights.
    seq_weights and number_of_diff_aas are modified in-place/returned.
    """
    query_length = len(matrix)
    
    # Initialize
    for pos in range(query_length):
        number_of_diff_aas[pos] = 0.0
        
    for i in range(len(alignment_strings)):
        seq_weights[i] = 0.0
        
    # Tabulate # of unique amino acids at each position
    for pos in range(query_length):
        for aa_code in range(26):
            aa = chr(aa_code + ord('A'))
            if valid_amino_acid(aa) and matrix[pos][aa_code] > 0.0:
                number_of_diff_aas[pos] += 1.0

    tot = 0.0
    for seq_index in range(len(alignment_strings)):
        for pos in range(query_length):
            aa = alignment_strings[seq_index].sequence[pos]
            aa_index = aa_to_idx(aa)
            
            if valid_amino_acid(aa) and matrix[pos][aa_index] > 0.0:
                tmp = number_of_diff_aas[pos] * matrix[pos][aa_index]
                if tmp != 0:
                     seq_weights[seq_index] += 1.0 / tmp
        tot += seq_weights[seq_index]
        
    # Normalize
    if tot == 0:
        tot = 1.0 # Avoid div by zero
        
    new_tot_weight = 0.0
    for seq_index in range(len(alignment_strings)):
        seq_weights[seq_index] = (seq_weights[seq_index] / tot) * len(alignment_strings)
        new_tot_weight += seq_weights[seq_index]

def create_matrix(alignment_strings, query, seq_weights, matrix, tot_pos_weight):
    """
    Populates matrix with weighted counts.
    """
    query_len = len(query.sequence)
    
    # Zero out matrix first? Assumed passed in zeroed or accumulating. C++ accumulates (+=)
    
    for seq_index in range(len(alignment_strings)):
        for pos in range(query_len):
            aa = alignment_strings[seq_index].sequence[pos]
            if valid_amino_acid(aa):
                aa_index = aa_to_idx(aa)
                matrix[pos][aa_index] += seq_weights[seq_index]
                tot_pos_weight[pos] += seq_weights[seq_index]

def find_max_aa_in_matrix(matrix, max_aa_index):
    query_length = len(matrix)
    aas = 26
    
    for pos in range(query_length):
        max_aa = -1
        max_count = -1.0
        for aa_index in range(aas):
            if matrix[pos][aa_index] > max_count:
                max_aa = aa_index
                max_count = matrix[pos][aa_index]
        max_aa_index[pos] = max_aa

def calc_epsilon(weighted_matrix, max_aa_array, number_of_diff_aas, epsilon):
    query_length = len(weighted_matrix)
    
    for pos in range(query_length):
        if number_of_diff_aas[pos] == 1:
            epsilon[pos] = 0
        else:
            max_aa = max_aa_array[pos]
            sum_val = 0.0
            pos_tot = 0.0
            
            for aa_code in range(26):
                aa = chr(aa_code + ord('A'))
                if valid_amino_acid(aa):
                    aa_index = aa_code
                    rank = rank_matrix[max_aa][aa_index]
                    sum_val += float(rank) * weighted_matrix[pos][aa_index]
                    pos_tot += weighted_matrix[pos][aa_index]
            
            if pos_tot > 0:
                sum_val = sum_val / pos_tot
            epsilon[pos] = math.exp(sum_val)

def add_diric_values(count_col, diric_col):
    diri_comp_num = len(diri_altot)
    probn = [0.0] * diri_comp_num
    probj = [0.0] * diri_comp_num
    
    pos_count_tot = sum(count_col)
    
    # Compute Prob(n|j)
    for j in range(diri_comp_num):
        probn[j] = math.lgamma(pos_count_tot + 1.0) + math.lgamma(diri_altot[j])
        probn[j] -= math.lgamma(pos_count_tot + diri_altot[j])
        
        for aa_code in range(26):
            aa = chr(aa_code + ord('A'))
            if valid_amino_acid(aa):
                aa_index = aa_code
                tmp = math.lgamma(count_col[aa_index] + diri_alpha[j][aa_index])
                tmp -= math.lgamma(count_col[aa_index] + 1.0)
                tmp -= math.lgamma(diri_alpha[j][aa_index])
                probn[j] += tmp

    # Sum qk * p(n|k)
    denom = math.log(diri_q[0]) + probn[0]
    for j in range(1, diri_comp_num):
        tmp = math.log(diri_q[j]) + probn[j]
        denom = add_logs(denom, tmp)
        
    # Prob(j|n)
    for j in range(diri_comp_num):
        probj[j] = math.log(diri_q[j]) + probn[j] - denom
        
    totreg = 0.0
    for aa_code in range(26):
         aa = chr(aa_code + ord('A'))
         if valid_amino_acid(aa):
             aa_index = aa_code
             for j in range(len(probj)):
                 diric_col[aa_index] += math.exp(probj[j]) * diri_alpha[j][aa_index]
             totreg += diric_col[aa_index]
             
    # Normalize
    if totreg > 0:
        for aa_code in range(26):
             diric_col[aa_code] /= totreg

def calc_diri(count_matrix, diric_matrix):
    query_length = len(count_matrix)
    for pos in range(query_length):
        add_diric_values(count_matrix[pos], diric_matrix[pos])

def scale_matrix_to_max_aa(matrix, max_aa_array):
    query_length = len(matrix)
    aas = 26
    for pos in range(query_length):
        max_aa = max_aa_array[pos]
        max_aa_score = matrix[pos][max_aa]
        if max_aa_score != 0:
            for aa_index in range(aas):
                matrix[pos][aa_index] = matrix[pos][aa_index] / max_aa_score

def calc_sift_scores(alignment_strings, query, matrix, sift_scores):
    query_length = len(matrix)
    
    # Initialize
    max_aa_array = [0] * query_length
    number_of_diff_aas = [0.0] * query_length
    epsilon = [0.0] * query_length
    
    num_seqs = len(alignment_strings)
    raw_count_matrix = [[0.0]*26 for _ in range(query_length)]
    
    # Weights 1
    weights_1 = [1.0] * num_seqs
    aas_stored_at_each_pos = [0.0] * query_length
    
    create_matrix(alignment_strings, query, weights_1, raw_count_matrix, aas_stored_at_each_pos)
    
    # Seq Weights
    seq_weights = [0.0] * num_seqs
    calc_seq_weights(alignment_strings, matrix, seq_weights, number_of_diff_aas)
    
    # Weighted Matrix
    seq_weighted_matrix = [[0.0]*26 for _ in range(query_length)]
    tot_weights_each_pos = [0.0] * query_length
    
    create_matrix(alignment_strings, query, seq_weights, seq_weighted_matrix, tot_weights_each_pos)
    
    find_max_aa_in_matrix(seq_weighted_matrix, max_aa_array)
    
    calc_epsilon(seq_weighted_matrix, max_aa_array, number_of_diff_aas, epsilon)
    
    # Pseudo Diri
    diric_matrix = [[0.0]*26 for _ in range(query_length)]
    calc_diri(seq_weighted_matrix, diric_matrix)
    
    # Final Calculation
    for pos in range(query_length):
        for aa_code in range(26):
            sift_scores[pos][aa_code] = seq_weighted_matrix[pos][aa_code] + epsilon[pos] * diric_matrix[pos][aa_code]
            if tot_weights_each_pos[pos] + epsilon[pos] != 0:
                sift_scores[pos][aa_code] /= (tot_weights_each_pos[pos] + epsilon[pos])
                
    find_max_aa_in_matrix(sift_scores, max_aa_array)
    scale_matrix_to_max_aa(sift_scores, max_aa_array)

def print_matrix_original_format(matrix, filename):
    try:
        with open(filename, 'w') as fp:
            query_length = len(matrix)
            aas = 26
            
            fp.write("ID   UNK_ID; MATRIX\nAC   UNK_AC\nDE   UNK_DE\nMA   UNK_BL\n")
            fp.write(" ")
            for aa_index in range(aas):
                # ignore J O U
                if aa_index != 9 and aa_index != 14 and aa_index != 20:
                    fp.write(f" {chr(aa_index + ord('A'))}  ")
            fp.write(" *   -\n")
            
            for pos in range(query_length):
                for aa_index in range(aas):
                    if aa_index != 9 and aa_index != 14 and aa_index != 20:
                        fp.write(f" {matrix[pos][aa_index]:6.4f} ")
                fp.write(" 0.0000  0.0000\n")
            fp.write("//\n")
    except Exception as e:
        sys.stderr.write(f"Error writing matrix to {filename}: {e}\n")

# Main SIFT Prediction wrapper
def sift_predictions(alignment_strings, query, subst_path, sequence_identity, out_path, prediction_cutoff=TOLERANCE_PROB_THRESHOLD):
    # Only keep first kMaxSequences - 1
    # Note: alignment_strings includes filtered hits.
    
    if len(alignment_strings) > kMaxSequences - 1:
        alignment_strings[:] = alignment_strings[:kMaxSequences - 1]
        
    query_length = len(query.sequence)
    remove_seqs_percent_identical_to_query(query, alignment_strings, sequence_identity)
    
    # Add query to beginning
    # In Python, we can just insert the query object itself if it's a Chain
    alignment_strings.insert(0, query)
    
    matrix = [[0.0]*26 for _ in range(query_length)]
    sift_scores = [[0.0]*26 for _ in range(query_length)]
    
    # Initial weights 1 call to setup matrix size/basic needed for calc_seq_weights
    # Just creating a temporary "matrix" as done in C++ call "createMatrix... matrix, aas_stored"
    weights_1_temp = [1.0] * len(alignment_strings)
    aas_stored_temp = [0.0] * query_length
    create_matrix(alignment_strings, query, weights_1_temp, matrix, aas_stored_temp)
    
    calc_sift_scores(alignment_strings, query, matrix, sift_scores)
    
    # Output
    out_extension = ".SIFTprediction"
    out_file_name = create_file_name(query.header.split()[0], out_path, out_extension) # Use first part of header as name
    
    # Helper to clean name? C++ chainGetName behaves specific way.
    # Assuming header has ID.
    
    print_matrix_original_format(sift_scores, out_file_name)

