# Example: Full PolyPhen-2 ML pipeline in Python
# This script demonstrates how to extract features, export ARFF, and classify with WEKA

from polyphen.blast_wrapper import run_blastp
from polyphen.msa_wrapper import run_mafft
from polyphen.psic_wrapper import run_psic
from polyphen.arff_export import export_polyphen_arff
from polyphen.weka_wrapper import run_weka_classifier

# 1. Prepare variant features (normally extracted from pipeline)
# For demonstration, use dummy data. Replace with real feature extraction.
variants = [
    {
        'o_acc': 'P12345', 'o_pos': 100, 'o_aa1': 'A', 'o_aa2': 'V',
        'acc': 'P12345', 'pos': 100, 'aa1': 'A', 'aa2': 'V',
        'nt1': '?', 'nt2': '?',
        'site': 0, 'region': 0, 'phat': 0.0,
        'score_delta': 0.0, 'score1': 0.0, 'score2': 0.0,
        'num_obs': 0, 'ali_ide': 0.0, 'ali_len': 0.0,
        'acc_normed': 0.0, 'sec_str': 0, 'map_region': 0,
        'delta_volume': 0.0, 'delta_prop': 0.0, 'b_fact': 0.0,
        'h_bonds': 0.0, 'het_cont_ave_num': 0.0, 'het_cont_min_dist': 0.0,
        'inter_cont_ave_num': 0.0, 'inter_cont_min_dist': 0.0,
        'transversion': 0, 'cpg': 0, 'pfam_hit': 0, 'id_p_max': 0.0,
        'id_p_snp': 0.0, 'id_q_min': 0.0
        # ...add all required features as per mapping...
    }
]

# 2. List of features in order (must match model)
feature_names = [
    'o_acc','o_pos','o_aa1','o_aa2','acc','pos','aa1','aa2','nt1','nt2',
    'site','region','phat','score_delta','score1','score2','num_obs','ali_ide','ali_len',
    'acc_normed','sec_str','map_region','delta_volume','delta_prop','b_fact','h_bonds',
    'het_cont_ave_num','het_cont_min_dist','inter_cont_ave_num','inter_cont_min_dist',
    'transversion','cpg','pfam_hit','id_p_max','id_p_snp','id_q_min'
    # ...add all required features as per mapping...
]

# 3. Export ARFF
arff_path = 'output/polyphen2_features.arff'
os.makedirs('output', exist_ok=True)
export_polyphen_arff(variants, feature_names, arff_path)

# 4. Run WEKA classifier
weka_jar = '/path/to/weka.jar'  # Update this path
model_file = 'models/HumDiv.UniRef100.NBd.f11.model'  # Update if needed
output_file = 'output/polyphen2_predictions.txt'
run_weka_classifier(arff_path, model_file, weka_jar, output_file)

print(f'Predictions written to: {output_file}')
