import os
import re

CONFIG = {}

def load_config(pph_root=None):
    """
    Load configuration from .cnf files in the config directory.
    Mimics the Perl PPH::Config behavior.
    """
    global CONFIG
    
    if pph_root is None:
        # Deduce PPH root from current file location or env var
        if 'PPH' in os.environ:
            pph_root = os.environ['PPH']
        else:
            # Assuming this file is in python_impl/polyphen/config.py
            # and we want to go up to root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            pph_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
            os.environ['PPH'] = pph_root
            
    CONFIG['PPH'] = pph_root
    
    # Default config paths
    config_dir = os.path.join(pph_root, 'config')
    
    # Order of loading similar to Perl: 
    # 1. Dist config files
    # 2. User config overrides (omitted for now for simplicity, can add later)
    
    config_files = [
        'paths.cnf.dist',
        'databases.cnf.dist',
        'programs.cnf.dist',
        'options.cnf.dist',
        # 'programs_options.cnf.dist', # Often has specific flags
    ]
    
    for cnf_file in config_files:
        # Load .cnf if exists, else .cnf.dist
        # Actually config_files list has .dist extension. Let's try to remove it.
        base = cnf_file.replace('.dist', '')
        
        # Try base (user config)
        path = os.path.join(config_dir, base)
        if os.path.exists(path):
            _parse_config_file(path)
        else:
            # Try dist
            path_dist = os.path.join(config_dir, cnf_file)
            if os.path.exists(path_dist):
                _parse_config_file(path_dist)
            
    # Post-process config
    _expand_variables()

def _parse_config_file(path):
    """Parses a simple KEY = VALUE config file."""
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Remove comments at end of line
            if '#' in line:
                line = line.split('#', 1)[0].strip()
                
            if '=' in line:
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip()
                # Remove quotes if present
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                
                CONFIG[key] = val

def _expand_variables():
    """Expands $CONFIG{KEY} and $ENV{KEY} variables in values."""
    # Simple expansion loop (naive but works for most cases)
    # We repeat a few times to handle nested vars
    for _ in range(3):
        for key, val in CONFIG.items():
            if not isinstance(val, str): continue
            
            # Replace $CONFIG{...}
            matches = re.findall(r'\$CONFIG\{(\w+)\}', val)
            for m in matches:
                if m in CONFIG:
                    val = val.replace(f'$CONFIG{{{m}}}', CONFIG[m])
            
            # Replace $ENV{...}
            matches_env = re.findall(r'\$ENV\{(\w+)\}', val)
            for m in matches_env:
                if m in os.environ:
                    val = val.replace(f'$ENV{{{m}}}', os.environ[m])

            # Replace $PPH manually if used without brackets (rare but possible in sh)
            # but in these configs it usually uses $CONFIG{PPH}
            
            CONFIG[key] = val

# Helper access
def get(key, default=None):
    return CONFIG.get(key, default)
