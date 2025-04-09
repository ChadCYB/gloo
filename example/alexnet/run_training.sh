#!/bin/bash

# Create timestamped results directory
timestamp=$(date +"%Y%m%d_%H%M%S")
results_dir="results/$timestamp"
mkdir -p "$results_dir"

# Number of processes to run
num_processes=10

# Kill any existing alexnet_trainer processes
pkill -f alexnet_trainer

# Create a unique store path for this run
store_path="/tmp/gloo_store_$timestamp"

# Set environment variable for store path
export STORE_PATH="$store_path"

# Function to check if a process is still running
is_process_running() {
    local pid=$1
    kill -0 $pid 2>/dev/null
}

# Start multiple processes with a delay
pids=()
for i in $(seq 0 $((num_processes-1))); do
    # Run each process in the background and redirect output to the results directory
    ./build/alexnet_trainer $i $num_processes "config.json" > "$results_dir/rank_$i.log" 2>&1 &
    pids+=($!)
    sleep 1
done

# Wait for all processes to complete
for pid in "${pids[@]}"; do
    while is_process_running $pid; do
        sleep 1
    done
done

# Record configuration files
echo "=== Main Config (config.json) ===" > "$results_dir/config.log"
cat config.json >> "$results_dir/config.log"
echo -e "\n=== Topology Config (topology_config.json) ===" >> "$results_dir/config.log"
cat topology_config.json >> "$results_dir/config.log"

# Parse the log files using the Python script
python3 parse_logs.py "$results_dir"

# Generate heatmaps
python3 generate_heatmap.py "$results_dir"

echo "All files have been saved in $results_dir:"
ls -l "$results_dir" 