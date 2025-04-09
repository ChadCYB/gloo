#!/usr/bin/env python3
import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def read_matrix_from_file(file_path):
    """Read a matrix from a file and return it as a numpy array."""
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
        
        # Split content into rows
        rows = content.split('\n')
        matrix = []
        
        for row in rows:
            # Split each row into values and convert to float
            values = [float(x) for x in row.split()]
            matrix.append(values)
        
        return np.array(matrix)
    except Exception as e:
        print(f"Error reading matrix from {file_path}: {e}")
        return None

def generate_heatmaps(results_dir):
    """Generate heatmaps for bandwidth and traffic matrices."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        print(f"Generating heatmaps in: {results_dir}")

        # Set the style for better visualization
        plt.style.use('default')
        sns.set_theme(style="whitegrid")

        # Generate bandwidth heatmap
        bandwidth_file = os.path.join(results_dir, 'bandwidth_matrix.log')
        if os.path.exists(bandwidth_file):
            print("Generating bandwidth heatmap...")
            bandwidth_matrix = read_matrix_from_file(bandwidth_file)
            if bandwidth_matrix is not None:
                # Get minimum non-zero value for vmin
                non_zero_mask = bandwidth_matrix > 0
                if np.any(non_zero_mask):
                    vmin = np.min(bandwidth_matrix[non_zero_mask])
                else:
                    vmin = 0
                
                # Create mask for upper triangle
                mask = np.triu(np.ones_like(bandwidth_matrix))
                
                plt.figure(figsize=(12, 10))
                sns.heatmap(bandwidth_matrix, 
                          annot=True, 
                          fmt='.2f', 
                          cmap='YlOrRd',
                          vmin=vmin,
                          vmax=np.max(bandwidth_matrix),
                          square=True,
                          mask=mask,
                          cbar_kws={'label': 'Bandwidth (GB/s)'})
                
                # Remove the grid lines
                plt.grid(False)
                plt.title('Bandwidth Matrix (GB/s) - Lower Triangle', pad=20, fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, 'bandwidth_heatmap.jpg'), 
                          dpi=300, 
                          bbox_inches='tight')
                plt.close()
                print("Generated bandwidth heatmap")
            else:
                print("Failed to read bandwidth matrix")
        else:
            print(f"Bandwidth matrix file not found: {bandwidth_file}")

        # Generate traffic heatmaps
        print("Generating traffic heatmaps...")
        traffic_files = sorted([f for f in os.listdir(results_dir) 
                              if f.startswith('traffic_matrix_epoch_') and f.endswith('.log')])
        
        for traffic_file in traffic_files:
            # Extract epoch number from filename
            epoch_num = re.search(r'traffic_matrix_epoch_(\d+)\.log', traffic_file).group(1)
            print(f"Processing traffic matrix for epoch {epoch_num}...")
            
            traffic_matrix = read_matrix_from_file(os.path.join(results_dir, traffic_file))
            if traffic_matrix is not None:
                plt.figure(figsize=(12, 10))
                sns.heatmap(traffic_matrix, 
                          annot=True, 
                          fmt='.2f', 
                          cmap='YlOrRd',
                          vmin=0,
                          vmax=np.max(traffic_matrix),
                          square=True,
                          cbar_kws={'label': 'Traffic (MB)'})
                plt.title(f'Traffic Matrix - Epoch {epoch_num} (MB)', pad=20, fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f'traffic_epoch_{epoch_num}.jpg'), 
                          dpi=300, 
                          bbox_inches='tight')
                plt.close()
                print(f"Generated traffic heatmap for epoch {epoch_num}")
            else:
                print(f"Failed to read traffic matrix for epoch {epoch_num}")

    except Exception as e:
        print(f"Error generating heatmaps: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 generate_heatmap.py <results_dir>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)
    
    generate_heatmaps(results_dir)
    print("Heatmap generation completed") 