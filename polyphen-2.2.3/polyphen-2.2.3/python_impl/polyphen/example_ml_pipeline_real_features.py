# PolyPhen-2: Example pipeline with real feature names (order matches original model)
import os
from polyphen.arff_export import export_polyphen_arff
from polyphen.weka_wrapper import run_weka_classifier

# List of real PolyPhen-2 features (order is critical)
feature_names = [
    'o_acc', 'o_pos', 'o_aa1', 'o_aa2', 'snp_id', 'acc', 'pos', 'aa1', 'aa2', 'nt1', 'nt2',
    'prediction', 'based_on', 'effect', 'site', 'region', 'PHAT', 'dScore', 'Score1', 'Score2',
    'MSAv', 'Nobs', 'Nstruct', 'Nfilt', 'PDB_id', 'PDB_pos', 'PDB_ch', 'ident', 'length',
    'NormASA', 'SecStr', 'MapReg', 'dVol', 'dProp', 'B-fact', 'H-bonds', 'AveNHet', 'MinDHet',
    'AveNInt', 'MinDInt', 'AveNSit', 'MinDSit', 'Transv', 'CodPos', 'CpG', 'MinDJnc',
    'PfamHit', 'IdPmax', 'IdPSNP', 'IdQmin', 'cpg_var1_var2', 'cpg_transition',
    'charge_change', 'hydroph_change', 'delta_volume_new', 'delta_prop_new'
]

# Dummy variant with all features (replace with real feature extraction logic)
variants = [
    {k: '?' for k in feature_names}  # All features set to '?' (missing); fill with real values
]

arff_path = 'output/polyphen2_features_real.arff'
os.makedirs('output', exist_ok=True)
export_polyphen_arff(variants, feature_names, arff_path)

weka_jar = '/path/to/weka.jar'  # Update this path
model_file = 'models/HumDiv.UniRef100.NBd.f11.model'  # Update if needed
output_file = 'output/polyphen2_predictions_real.txt'
run_weka_classifier(arff_path, model_file, weka_jar, output_file)

print(f'Predictions written to: {output_file}')
