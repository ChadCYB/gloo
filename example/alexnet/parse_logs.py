#!/usr/bin/env python3

import re
import os
import sys

def extract_traffic_matrices(input_file, results_dir):
    """Extract traffic matrices from the log file and save them to separate files."""
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Find all traffic matrices
    traffic_matrices = re.findall(r'Traffic Matrix \(MB\):\n([\d\s.]+)', content)
    
    # Save each traffic matrix to a separate file
    for i, matrix in enumerate(traffic_matrices, 1):
        with open(f"{results_dir}/traffic_matrix_epoch_{i}.log", 'w') as f:
            f.write(matrix.strip())
            print(f"Saved traffic matrix for epoch {i}")

def extract_bandwidth_matrix(input_file, results_dir):
    """Extract bandwidth matrix from the log file and save it to a separate file."""
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Find all bandwidth measurements
    bandwidth_pattern = r"Bandwidth between GPU (\d+) and GPU (\d+): ([\d.]+) GB/s"
    bandwidth_matches = re.findall(bandwidth_pattern, content)
    
    # Create a 10x10 matrix initialized with zeros
    bandwidth_matrix = [[0.0 for _ in range(10)] for _ in range(10)]
    
    # Fill in the bandwidth values
    for src, dst, bw in bandwidth_matches:
        src = int(src)
        dst = int(dst)
        bandwidth_matrix[src][dst] = float(bw)
        bandwidth_matrix[dst][src] = float(bw)  # Symmetric
    
    # Save the bandwidth matrix to a file
    with open(f"{results_dir}/bandwidth_matrix.log", 'w') as f:
        for row in bandwidth_matrix:
            f.write(' '.join(f"{val:.3f}" for val in row) + '\n')
        print("Saved bandwidth matrix")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 parse_logs.py <results_dir>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    input_file = f"{results_dir}/rank_0.log"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} does not exist")
        sys.exit(1)
    
    extract_traffic_matrices(input_file, results_dir)
    extract_bandwidth_matrix(input_file, results_dir)

if __name__ == "__main__":
    main() 