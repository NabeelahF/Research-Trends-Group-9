
DATA_PREDICTIONS = ['unknown', 'benign', 'possibly damaging', 'probably damaging']

def test(protein, snp):
    """
    Rule based prediction.
    Ported from PPH::Rules::test
    """
    
    # Extract features
    features = snp.features
    
    # Alignment scores
    selected = snp.scores.get('Selected', '')
    delta = None
    if selected and selected in snp.scores:
         try:
             # PsicD might be string "+1.234"
             psic_d_str = snp.scores[selected].get('PsicD')
             if psic_d_str:
                 delta = abs(float(psic_d_str))
         except ValueError:
             pass
             
    # Structure
    structure = snp.structure.get('HitsMapped') if snp.structure else None
    struct_params = None
    struct_contact = None
    
    if structure:
        # Assuming we parsed structure maps. 
        # For now, structural logic relies on data we might not have fully populated yet.
        # But let's implement the logic assuming the data is there.
        if 'Maps' in snp.structure and snp.structure['Maps']:
             struct_params = snp.structure['Maps'][0].get('Params')
        
        struct_contact = snp.structure.get('BestContact')

    avail = []
    if features: avail.append('FT')
    if delta is not None: avail.append('alignment')
    if structure: avail.append('structure')
    
    if not avail:
        return {
            'Prediction': DATA_PREDICTIONS[0], # unknown
            'Avail': [],
            'Basis': None,
            'Effect': None,
            'Data': None
        }

    # 1. UniProt Site features
    if 'Site' in features:
        site = features['Site']
        if site != 'CARBOHYD':
             # BONDS check omitted for brevity, assuming standard non-bond site is 2.2
             return {
                 'Prediction': DATA_PREDICTIONS[3], # properly damaging
                 'Avail': avail,
                 'Basis': 'sequence annotation',
                 'Effect': '2.2', # simplified
                 'Data': f"Site type: {site}"
             }
             
    # 2. Transmembrane
    if 'Region' in features and features['Region'] == 'TRANSMEM':
         phat = features.get('PHAT', 0)
         pred = DATA_PREDICTIONS[2] if phat < 0 else DATA_PREDICTIONS[1] # possibly vs benign
         return {
             'Prediction': pred,
             'Avail': avail,
             'Basis': 'sequence annotation',
             'Effect': '2.2.2' if phat < 0 else None,
             'Data': f"PHAT matrix element difference: {phat}"
         }

    # If nothing else..
    if delta is None:
         return {
            'Prediction': DATA_PREDICTIONS[0],
            'Avail': avail,
            'Basis': None,
            'Effect': None,
            'Data': None
        }

    basis = 'alignment'
    if selected == 'Mz': basis += '_mz'

    # 3. Structure contacts
    if delta > 1 and structure and struct_contact:
         # A) Heteroatoms
         # B) Critical sites
         # Simplified check
         pass

    # 4. PSIC + Structure parameters
    if delta <= 0.5:
         return {
            'Prediction': DATA_PREDICTIONS[1], # benign
            'Avail': avail,
            'Basis': basis,
            'Effect': None,
            'Data': f"PSIC score difference: {delta}"
        }
        
    if 0.5 < delta <= 1.5:
        # Structure logic check would go here
        # ...
        pass
        
        # Fallback
        return {
            'Prediction': DATA_PREDICTIONS[1], # benign
            'Avail': avail,
            'Basis': basis,
            'Effect': None,
            'Data': f"PSIC score difference: {delta}"
        }
        
    if 1.5 < delta <= 2.0:
         # Structure logic
         # ...
         pass
         
         return {
            'Prediction': DATA_PREDICTIONS[2], # possibly damaging
            'Avail': avail,
            'Basis': basis,
            'Effect': None,
            'Data': f"PSIC score difference: {delta}"
        }
        
    if delta > 2.0:
        return {
            'Prediction': DATA_PREDICTIONS[3], # probably damaging
            'Avail': avail,
            'Basis': basis,
            'Effect': None,
            'Data': f"PSIC score difference: {delta}"
        }
        
    return {
            'Prediction': DATA_PREDICTIONS[0],
            'Avail': avail,
            'Basis': None,
            'Effect': None,
            'Data': None
        }
