import sys
import os
import argparse

# Add package to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from polyphen import pipeline

def main():
    parser = argparse.ArgumentParser(description="PolyPhen-2 Python Port")
    parser.add_argument('input_file', help="Input file with protein substitutions")
    parser.add_argument('-o', '--output', help="Output file (default: stdout)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)
        
    print(f"Processing {args.input_file}...")
    results = pipeline.run_analysis(args.input_file)
    
    output_text = pipeline.format_output(results)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_text)
        print(f"Results written to {args.output}")
    else:
        print(output_text)

if __name__ == "__main__":
    main()
