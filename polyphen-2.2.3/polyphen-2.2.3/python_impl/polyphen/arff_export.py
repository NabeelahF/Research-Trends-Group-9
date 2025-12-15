import os

def export_polyphen_arff(variants, feature_names, arff_path, relation_name="polyphen2"):
    """
    Export PolyPhen-2 features to an ARFF file for WEKA.
    Args:
        variants (list of dict): List of feature dicts for each variant.
        feature_names (list of str): Ordered list of feature names (matching model).
        arff_path (str): Output ARFF file path.
        relation_name (str): ARFF relation name.
    Returns:
        None
    """
    with open(arff_path, 'w') as f:
        f.write(f"@RELATION {relation_name}\n\n")
        for feat in feature_names:
            # For simplicity, treat all as numeric or string; adjust as needed
            f.write(f"@ATTRIBUTE {feat} NUMERIC\n")
        f.write("@ATTRIBUTE class {benign,possibly_damaging,probably_damaging,unknown}\n\n")
        f.write("@DATA\n")
        for var in variants:
            row = []
            for feat in feature_names:
                val = var.get(feat, '?')
                if val is None:
                    val = '?'
                row.append(str(val))
            # Add dummy class label (unknown) for prediction
            row.append('unknown')
            f.write(",".join(row) + "\n")
    print(f"ARFF exported: {arff_path}")
