#!/usr/bin/env python3
"""
Set the MSA binary path in config/programs.cnf (created from programs.cnf.dist).
Usage:
  python python_impl/set_msa_path.py <path-to-mafft-or-muscle-or-name-in-PATH>
Example:
  python python_impl/set_msa_path.py mafft
  python python_impl/set_msa_path.py C:\\tools\\mafft\\bin\\mafft.exe
"""
import sys
import os

def main():
    if len(sys.argv) < 2:
        print('Usage: set_msa_path.py <msa-path-or-exe-name>')
        return 2
    msa_path = sys.argv[1]
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_dir = os.path.join(repo_root, 'config')
    dist_file = os.path.join(config_dir, 'programs.cnf.dist')
    out_file = os.path.join(config_dir, 'programs.cnf')

    if not os.path.exists(dist_file):
        print('Error: programs.cnf.dist not found at', dist_file)
        return 3

    with open(dist_file, 'r', encoding='utf-8') as fh:
        lines = fh.readlines()

    new_lines = []
    replaced = False
    for line in lines:
        if line.strip().startswith('MSA') and '=' in line:
            new_lines.append(f"MSA             = {msa_path}\n")
            replaced = True
        else:
            new_lines.append(line)

    if not replaced:
        # append MSA line
        new_lines.append(f"\n# Added by set_msa_path.py\nMSA             = {msa_path}\n")

    with open(out_file, 'w', encoding='utf-8') as fh:
        fh.writelines(new_lines)

    print('Wrote', out_file)
    print('MSA set to:', msa_path)
    return 0

if __name__ == '__main__':
    sys.exit(main())
