import subprocess
import os
import sys
from .config import get as config_get

def run_command(cmd, args, cwd=None, env=None):
    """
    Run an external command.
    """
    full_cmd = [cmd] + args
    print(f"Running: {' '.join(full_cmd)}")
    
    try:
        # Check if cmd is a path or just a name
        executable = cmd
        if not os.path.exists(executable) and os.path.sep not in executable:
             # Assume it's in path
             pass 
        
        result = subprocess.run(
            full_cmd, 
            cwd=cwd, 
            env=env,
            check=True,
            capture_output=True,
            text=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(f"Stderr: {e.stderr}")
        raise

def locate_file(filename, *directories):
    """
    Look for a file in a list of directories.
    """
    for d in directories:
        if not d: continue
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return path
    return None

def check_binary(config_key):
    """
    Get binary path from config and check if it exists.
    """
    path = config_get(config_key)
    if not path:
        return None
        
    # Remove trailing slash if it mistakenly exists for a binary
    if path.endswith(os.path.sep):
        path = path[:-1]
        
    if os.path.exists(path) or os.path.sep not in path:
        return path

    # Windows compatibility: try appending .exe
    if os.name == 'nt' and not path.lower().endswith('.exe'):
        path_exe = path + '.exe'
        if os.path.exists(path_exe):
            return path_exe
    
    print(f"Warning: Binary {config_key} not found at {path}")
    return path
