class Protein:
    def __init__(self, acc, seq=None, name=None, desc=None):
        self.acc = acc
        self.seq = seq
        self.name = name
        self.desc = desc
        self.length = len(seq) if seq else 0
        self.features = {} # UniProt features
        self.profile = {}  # PSIC profile data

class SNP:
    def __init__(self, acc, pos, aa1, aa2, comments=None):
        self.acc = acc # Protein Accession
        self.pos = int(pos) # 1-based position
        self.aa1 = aa1 # Ref AA
        self.aa2 = aa2 # Var AA
        self.comments = comments
        
        self.scores = {} # Place to store PSIC scores
        self.features = {} # Site features
        self.structure = {} # Structural data
        self.prediction = {} # Final prediction results
    
    def __repr__(self):
        return f"SNP({self.acc}: {self.aa1}{self.pos}{self.aa2})"
