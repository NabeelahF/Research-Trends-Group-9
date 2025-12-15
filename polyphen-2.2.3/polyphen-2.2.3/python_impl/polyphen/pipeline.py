from . import config, objects, profile, rules

def run_analysis(input_file):
    """
    Main analysis pipeline.
    Reads input, processes each variant, prints results.
    """
    # Initialize basic config
    config.load_config()
    
    results = []
    
    with open(input_file, 'r') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split()
            # Expect: Acc Pos Aa1 Aa2
            if len(parts) >= 4:
                acc = parts[0]
                pos = parts[1]
                aa1 = parts[2]
                aa2 = parts[3]
                
                # Check for comment
                comment = None
                if '#' in line:
                    comment = line.split('#', 1)[1].strip()
                
                snp = objects.SNP(acc, pos, aa1, aa2, comments=comment)
                
                # Create protein stub
                # In a real run, we'd fetch sequence here
                # For this port, we lazily rely on 'profile.py' logic which currently 
                # assumes we might generate seq file or it already exists?
                # Actually profile logic needs a Sequence.
                # Let's try to load sequence if possible, or just pass the stub.
                protein = objects.Protein(acc)
                
                # 1. Profile Calculation / Loading
                try:
                    profile.set_PSIC_profile(protein, snp)
                except Exception as e:
                    print(f"Error processing profile for {acc}: {e}")
                    
                # 2. Features (Skipped for now - normally uses SQL/Uniprot)
                
                # 3. Apply Rules
                prediction = rules.test(protein, snp)
                snp.prediction = prediction
                
                results.append(snp)
            else:
                print(f"Skipping invalid line {line_no}: {line}")
                
    return results

def format_output(results):
    """
    Format results as tab-delimited text matching PPH2 output format.
    Columns 0-52 roughly.
    """
    # Based on PPH.pm and pph2arff.pl expectation
    # 0: o_acc, 1: o_pos, 2: o_aa1, 3: o_aa2
    # 4: snp_id/rsid
    # 5: acc, 6: pos, 7: aa1, 8: aa2
    # 9: nt1, 10: nt2
    # 11: prediction
    # 12: based_on
    # 13: effect
    # 14: site, 15: region, 16: PHAT
    # 17: dScore, 18: Score1, 19: Score2
    # 20: MSAv, 21: Nobs
    # 22: Nstruct, 23: Nfilt
    # 24: PDB_id, 25: PDB_pos, 26: PDB_ch
    # 27: ident, 28: length
    # 29: NormASA, 30: SecStr, 31: MapReg
    # 32: dVol, 33: dProp
    # 34: B-fact, 35: H-bonds
    # 36: AveNHet, 37: MinDHet
    # 38: AveNInt, 39: MinDInt
    # 40: AveNSit, 41: MinDSit
    # 42: Transv, 43: CodPos, 44: CpG, 45: MinDJnc
    # 46: PfamHit, 47: IdPmax, 48: IdPSNP, 49: IdQmin
    
    header = [
        "o_acc", "o_pos", "o_aa1", "o_aa2", 
        "rsid", 
        "acc", "pos", "aa1", "aa2", 
        "nt1", "nt2", 
        "prediction", "based_on", "effect", 
        "site", "region", "PHAT", 
        "dScore", "Score1", "Score2", 
        "MSAv", "Nobs", 
        "Nstruct", "Nfilt", 
        "PDB_id", "PDB_pos", "PDB_ch", 
        "ident", "length", 
        "NormASA", "SecStr", "MapReg", 
        "dVol", "dProp", 
        "B-fact", "H-bonds", 
        "AveNHet", "MinDHet", 
        "AveNInt", "MinDInt", 
        "AveNSit", "MinDSit", 
        "Transv", "CodPos", "CpG", "MinDJnc", 
        "PfamHit", "IdPmax", "IdPSNP", "IdQmin",
        "Comments" # Extra
    ]
    
    lines = ["\t".join(header)]
    
    for snp in results:
        # Defaults
        vals = [""] * 52 # Ensure enough columns
        
        # Original & Mapped (assuming same for now)
        vals[0] = snp.acc
        vals[1] = str(snp.pos)
        vals[2] = snp.aa1
        vals[3] = snp.aa2
        
        vals[4] = "" # rsid
        
        vals[5] = snp.acc
        vals[6] = str(snp.pos)
        vals[7] = snp.aa1
        vals[8] = snp.aa2
        
        # Nucleotides (placeholder)
        vals[9] = "" 
        vals[10] = ""
        
        # Prediction
        vals[11] = snp.prediction.get('Prediction', 'unknown')
        vals[12] = snp.prediction.get('Basis', '') or ''
        vals[13] = str(snp.prediction.get('Effect', '')) if snp.prediction.get('Effect') else ''
        
        # Features
        feat = snp.features # dict
        vals[14] = feat.get('Site', '')
        vals[15] = feat.get('Region', '')
        vals[16] = str(feat.get('PHAT', ''))
        
        # Scores
        msa = snp.scores.get('Msa', {})
        if msa:
            vals[17] = f"{msa.get('PsicD', 0):.3f}"
            vals[18] = f"{msa.get('Psic1', 0):.3f}"
            vals[19] = f"{msa.get('Psic2', 0):.3f}"
            vals[21] = str(msa.get('Nobs', 0))
            
        # Placeholders for Structure, Pfam, etc.
        # Ideally we'd map snp.structure info here
        
        # Add comment at end using #
        row_str = "\t".join(str(x) for x in vals)
        if snp.comments:
            row_str += f"\t# {snp.comments}"
            
        lines.append(row_str)
        
    return "\n".join(lines)
