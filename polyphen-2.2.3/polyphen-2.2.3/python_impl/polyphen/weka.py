import os
import math
import subprocess
from .config import get as config_get
from .utils import run_command

# Constants for derived features (from pph2arff.pl)
VOLUME = {
  'A': 88, 'C': 108, 'D': 111, 'E': 138, 'F': 190,
  'G': 60, 'H': 153, 'I': 167, 'K': 168, 'L': 167,
  'M': 163,'N': 114, 'P': 112, 'Q': 144, 'R': 173,
  'S': 89, 'T': 116, 'V': 140, 'W': 227, 'Y': 193
}

PROPENSITY = {
  'A': [  0.59,   0.06,  -0.03,  -0.20,  -0.23,  -0.18,   0.18 ],
  'R': [ -2.99,  -1.14,   0.02,   0.33,   0.47,   0.25,  -0.31 ],
  'N': [ -1.48,  -0.55,  -0.27,   0.06,   0.26,   0.45,   0.39 ],
  'D': [ -1.35,  -0.69,  -0.33,   0.10,   0.28,   0.42,   0.50 ],
  'C': [  0.61,   0.52,   0.51,  -0.02,  -0.75,  -1.80,  -2.00 ],
  'Q': [ -1.55,  -0.58,  -0.25,   0.06,   0.40,   0.44,  -0.17 ],
  'E': [ -2.13,  -0.98,  -0.35,  -0.01,   0.40,   0.58,   0.28 ],
  'G': [  0.31,  -0.23,  -0.07,  -0.21,  -0.12,   0.05,   0.68 ],
  'H': [ -1.09,  -0.11,   0.25,   0.38,   0.10,  -0.18,  -0.54 ],
  'I': [  0.87,   0.54,   0.09,  -0.04,  -0.79,  -1.29,  -2.00 ],
  'L': [  0.59,   0.61,   0.28,  -0.11,  -0.76,  -1.21,  -1.42 ],
  'K': [ -4.06,  -2.28,  -1.22,  -0.08,   0.60,   0.75,   0.39 ],
  'M': [  0.53,   0.61,  -0.01,  -0.11,  -0.40,  -1.17,  -0.62 ],
  'F': [  0.47,   0.64,   0.36,  -0.01,  -0.86,  -1.31,  -1.75 ],
  'P': [ -0.98,  -0.46,  -0.28,  -0.03,   0.11,   0.41,   0.74 ],
  'S': [ -0.29,  -0.21,   0.04,  -0.11,  -0.01,   0.21,   0.49 ],
  'T': [ -0.49,  -0.16,  -0.11,   0.10,   0.23,   0.09,  -0.15 ],
  'W': [ -0.29,   0.53,   0.63,   0.19,  -0.52,  -1.42,  -2.54 ],
  'Y': [ -0.76,   0.27,   0.53,   0.43,  -0.22,  -0.95,  -1.87 ],
  'V': [  0.69,   0.50,   0.19,  -0.13,  -0.46,  -1.13,  -1.76 ]
}

HYDROPHOBICITY = {
  'A': 0.02, 'R': -0.42, 'N': -0.77, 'D': -1.04,  'C':  0.77, 
  'Q': -1.10, 'E': -1.14, 'G': -0.80, 'H': 0.26,  'I': 1.81,
  'L': 1.14, 'K': -0.41, 'M':  1.00, 'F':  1.35,  'P': -0.09, 
  'S': -0.97, 'T': -0.77, 'W':  1.71, 'Y': 1.11,  'V': 1.13
}

# Feature Attributes for ARFF header (f11)
ATTRIBUTES_F11 = [
    ('score_delta', 'numeric'),
    ('score1', 'numeric'),
    ('num_obs', 'numeric'),
    ('pfam_hit', ['NO', 'YES']),
    ('id_p_max', 'numeric'),
    ('id_q_min', 'numeric'),
    ('cpg_transition', ['NO', 'YES_TRANSITION', 'YES_TRANSVERSION']),
    ('delta_prop_new', 'numeric'),
    ('acc_normed', 'numeric'),
    ('b_fact', 'numeric'),
    ('class', ['NEUTRAL', 'DELETERIOUS'])
]

# Indices in PPH2 output (0-based) that map to f11 features
# Based on PPH2WEKA mapping in pph2arff.pl, column Pos11
# score_delta  -> dScore (17)
# score1       -> Score1 (18)
# num_obs      -> Nobs (21)
# pfam_hit     -> PfamHit (46)
# id_p_max     -> IdPmax (47)
# id_q_min     -> IdQmin (49)
# cpg_transition -> Derived
# delta_prop_new -> Derived
# acc_normed   -> NormASA (29)
# b_fact       -> B-fact (34)

def to_arff(input_file, output_file, mode='f11'):
    """
    Convert PPH2 output file to ARFF format.
    """
    data_rows = []
    
    with open(input_file, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split('\t')
            if len(parts) < 50: continue # Should be at least 50 cols
            
            # Skip unknown if dScore is empty
            prediction = parts[11]
            dScore = parts[17]
            if prediction == 'unknown' and not dScore:
                continue
                
            aa1 = parts[7]
            aa2 = parts[8]
            
            # Calculate Derived Features needed for F11
            # 1. cpg_transition
            cpg = parts[44]
            transv = parts[42]
            cpg_transition = 'NO'
            if cpg == 'YES':
                cpg_transition = 'YES_' + ('TRANSVERSION' if transv == 'YES' else 'TRANSITION')
            
            # 2. delta_prop_new
            # Depends on NormASA (29) and dProp (33)
            # But dProp must be calculated if not present?
            # pipeline.py might leave them empty.
            norm_asa = parts[29]
            d_prop = parts[33]
            
            # If dProp is empty, calculating...
            def _safe_float(s, default=0.0):
                try:
                    if s is None:
                        return default
                    ss = str(s).strip()
                    if ss == '' or ss == '?':
                        return default
                    return float(ss)
                except Exception:
                    return default

            if not d_prop or d_prop == '?':
                if aa1 in PROPENSITY and aa2 in PROPENSITY:
                    # Need interval j based on norm_asa (use robust numeric parsing)
                    x = _safe_float(norm_asa, 0)
                    j = 0
                    if 0 < x < 5:
                        j = 1
                    elif 5 <= x < 15:
                        j = 2
                    elif 15 <= x < 30:
                        j = 3
                    elif 30 <= x < 50:
                        j = 4
                    elif 50 <= x < 75:
                        j = 5
                    elif x >= 75:
                        j = 6

                    dp = PROPENSITY[aa1][j] - PROPENSITY[aa2][j]
                    d_prop = abs(dp)
                else:
                    d_prop = 0

            delta_prop_new = d_prop
            if _safe_float(norm_asa, 0) > 0.05:  # ASA_THRESH
                delta_prop_new = 0
            
            # Gather F11 Data
            row = []
            
            # Helper for numeric
            def get_num(idx, default=0):
                val = parts[idx]
                if not val or val == '?': return default
                try: return float(val)
                except: return default
                
            # Helper for yes/no/nominal
            def get_nom(idx, default='NO'):
                val = parts[idx]
                if not val or val == '?': return default
                return val

            # Features in order
            row.append(get_num(17)) # score_delta
            row.append(get_num(18)) # score1
            row.append(get_num(21)) # num_obs
            row.append(get_nom(46)) # pfam_hit
            row.append(get_num(47)) # id_p_max
            row.append(get_num(49)) # id_q_min
            row.append(cpg_transition) # cpg_transition
            row.append(delta_prop_new) # delta_prop_new
            row.append(get_num(29)) # acc_normed
            row.append(get_num(34)) # b_fact
            
            # Class (Unknown/Neutral for test, usually input output doesn't have it unless training)
            row.append('NEUTRAL') 
            
            data_rows.append(row)
            
    # Write ARFF
    with open(output_file, 'w') as out:
        out.write(f"@relation PPHv2.f11\n\n")
        
        for name, type_def in ATTRIBUTES_F11:
            if isinstance(type_def, list):
                out.write(f"@attribute {name} {{{','.join(type_def)}}}\n")
            else:
                out.write(f"@attribute {name} {type_def}\n")
                
        out.write("\n@data\n")
        for row in data_rows:
            # Join fields
            line = []
            for item in row:
                if isinstance(item, float):
                    line.append(f"{item:.4g}")
                else:
                    line.append(str(item))
            out.write(",".join(line) + "\n")

    return len(data_rows)

def run_classifier(arff_file, model_file, weka_jar_path):
    """
    Run Weka classifier.
    Returns list of (predicted_class, probability)
    """
    
    cmd = [
        "java", "-Xmx1024m",
        "-cp", weka_jar_path,
        "weka.classifiers.bayes.NaiveBayes",
        "-l", model_file,
        "-o", "-p", "0", # output predictions
        "-T", arff_file
    ]
    
    try:
        output = run_command("java", cmd[1:]) # reusing utils which runs subprocess
    except Exception as e:
        print(f"Weka execution failed: {e}")
        return []

    results = parse_weka_output(output)
    return results

def parse_weka_output(output):
    """
    Parse Weka prediction output.
    """
    results = []
    # Lines look like: 
    # inst# actual predicted error prediction
    # 1 1:NEUTRAL 1:NEUTRAL       0.999
    
    lines = output.splitlines()
    start = False
    for line in lines:
        line = line.strip()
        if not line: continue
        if "Predictions on test data" in line:
            start = True
            continue
        if not start: continue
        if line.startswith("inst#"): continue
        
        parts = line.split()
        if len(parts) >= 4:
            # Parse prediction column: "2:DELETERIOUS" or "1:NEUTRAL"
            pred_col_raw = parts[2]
            
            # Prob defaults to the last column usually
            # But parsing depends on if there is a '+' error column
            # Weka output: inst, actual, predicted, error(opt), probability
            
            prob_val = 0.0
            pred_class = 'neutral'
            
            if ":" in pred_col_raw:
                pred_class_str = pred_col_raw.split(":")[1]
                if "DELETERI" in pred_class_str: pred_class = 'deleterious'
                elif "NEUTRAL" in pred_class_str: pred_class = 'neutral'
            
            # Probability is usually last
            try:
                prob_val = float(parts[-1])
            except:
                prob_val = 0.0
                
            # Weka output probability of the *predicted* class.
            # We want probability of DELETERIOUS.
            # If predicted is NEUTRAL, prob(DEL) = 1 - prob(NEUTRAL)
            if pred_class == 'neutral':
                prob_val = 1.0 - prob_val
                
            results.append((pred_class, prob_val))
            
    return results
