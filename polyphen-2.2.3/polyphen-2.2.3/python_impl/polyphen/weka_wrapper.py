import subprocess
import os

def run_weka_classifier(arff_file, model_file, weka_jar_path, output_file=None, classifier="weka.classifiers.bayes.NaiveBayes"): 
    """
    Run WEKA classifier on an ARFF file using a pre-trained model.
    Args:
        arff_file (str): Path to input ARFF file (features).
        model_file (str): Path to WEKA binary model file.
        weka_jar_path (str): Path to weka.jar.
        output_file (str): Path to write classifier output (optional).
        classifier (str): WEKA classifier class name.
    Returns:
        str: Classifier output (if output_file is None).
    """
    cmd = [
        'java', '-Xmx1024m', '-cp', weka_jar_path,
        classifier,
        '-l', model_file,
        '-o', '-p', '0',
        '-T', arff_file
    ]
    print(f"Running WEKA: {' '.join(cmd)}")
    if output_file:
        with open(output_file, 'w') as out_f:
            result = subprocess.run(cmd, stdout=out_f, stderr=subprocess.PIPE, text=True)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"WEKA failed: {result.stderr}")
        raise RuntimeError(f"WEKA failed: {result.stderr}")
    if not output_file:
        return result.stdout
    print(f"WEKA finished. Output: {output_file}")
