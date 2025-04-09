# Distributed AlexNet Training with Gloo

This example demonstrates distributed training of AlexNet using the Gloo communication library.

## Features

- Distributed training across multiple GPUs
- Flexible topology support (ring and hierarchical)
- Configurable ring permutations for optimized communication
- Bandwidth measurement and visualization
- Configurable training parameters
- Realistic bandwidth simulation with configurable limits
- Buffer overflow detection and prevention

## Project Structure

- `alexnet_trainer.cpp`: Main training implementation
  - `DistributedTrainer` class: Handles distributed training across multiple GPUs
  - Measures bandwidth between GPUs
  - Tracks traffic during reduce-scatter and all-gather operations
  - Implements parameter synchronization with topology-aware data distribution
  - Logs training metrics and performance data

- `topology.h`: Topology management
  - Supports multiple topology types (ring, hierarchical)
  - Configurable ring permutations
  - Manages reduce-scatter and all-gather connections
  - Handles data size distribution across rings

- `alexnet_cuda.h`: CUDA implementation of AlexNet
  - Neural network architecture definition
  - GPU-accelerated forward and backward passes
  - Parameter management and synchronization

- `config_parser.h`: Configuration management
  - Parses JSON configuration file
  - Validates training parameters
  - Manages distributed settings
  - Handles logging configuration

- `generate_heatmap.py`: Visualization tool
  - Parses training metrics
  - Generates heatmaps of GPU-to-GPU bandwidth
  - Creates separate traffic visualizations for each epoch
  - Saves all outputs to the results directory

- `config.json`: Configuration file
  - Training parameters (epochs, steps, learning rate)
  - Distributed settings (nodes, host, port)
  - Topology configuration
  - Logging configuration

## Configuration

The training can be configured through `config.json`:

```json
{
    "model": {
        "input_channels": 3,
        "num_classes": 10
    },
    "training": {
        "batch_size": 32,
        "num_epochs": 5,
        "steps_per_epoch": 100,
        "learning_rate": 0.001
    },
    "distributed": {
        "num_nodes": 10,
        "use_localhost": true,
        "host": "127.0.0.1",
        "port": 29500,
        "bandwidth_limit": 10.0
    },
    "topology": {
        "type": "ring",
        "num_nodes": 10,
        "permutations": [0, 5]
    },
    "logging": {
        "log_dir": "results",
        "enable_metrics": true
    }
}
```

### Configuration Parameters

- `model.input_channels`: Number of input channels (3 for RGB images)
- `model.num_classes`: Number of output classes
- `training.batch_size`: Batch size for training
- `training.num_epochs`: Number of training epochs
- `training.steps_per_epoch`: Number of steps per epoch
- `training.learning_rate`: Learning rate for optimization
- `distributed.num_nodes`: Number of nodes in the distributed system
- `distributed.use_localhost`: Whether to use localhost for communication
- `distributed.host`: Host address for communication
- `distributed.port`: Port number for communication
- `distributed.bandwidth_limit`: Maximum bandwidth between nodes in GB/s
- `topology.type`: Type of topology ("ring" or "hierarchical")
- `topology.num_nodes`: Number of nodes in the topology
- `topology.permutations`: List of ring permutations for ring topology
- `logging.log_dir`: Directory for storing logs and metrics
- `logging.enable_metrics`: Whether to enable metrics collection

## Prerequisites

- CUDA toolkit (tested with version 12.0)
- CMake (version 3.10 or higher)
- Python 3 with required packages:
  - numpy
  - matplotlib
  - seaborn

## Building and Running

1. Build the project:
```bash
mkdir build && cd build
cmake ..
make -j
```

2. Run the training:
```bash
./run_training.sh
```

The script will:
- Start the specified number of training processes
- Measure bandwidth between nodes
- Run the training with the configured parameters
- Generate bandwidth and traffic heatmaps

## Topology Implementation

The system supports two topology types:

1. **Ring Topology**:
   - Supports multiple ring permutations
   - Each ring handles a portion of the data
   - Optimized for reduce-scatter and all-gather operations
   - Configurable rotation for each ring

2. **Hierarchical Topology**:
   - Two-level hierarchy
   - Intra-level and inter-level connections
   - Optimized for hierarchical communication patterns

## Parameter Synchronization

The parameter synchronization process:
1. Divides parameters among available rings
2. Performs reduce-scatter operation
3. Performs all-gather operation
4. Monitors buffer usage to prevent overflows
5. Logs traffic patterns for analysis

## Output

The training generates:
- Bandwidth matrix showing measured bandwidth between nodes
- Traffic matrix showing data transfer between nodes
- Heatmap visualizations of both matrices
- Training metrics and logs in the specified directory

## Notes

- The bandwidth limit parameter helps simulate real-world network constraints
- The actual training time will be affected by the bandwidth limit
- Higher bandwidth limits will result in faster training but less realistic simulation
- The default bandwidth limit is 10 GB/s, which is typical for high-speed interconnects
- Buffer overflow detection helps prevent memory issues during training

## Output Files

All output files are saved in the `results` directory:

- `training_metrics.txt`: Contains detailed training metrics including:
  - Configuration settings
  - Bandwidth measurements between GPUs
  - Traffic patterns during reduce-scatter and all-gather operations
  - Training time and performance statistics
  - Buffer overflow warnings (if any)

- `bandwidth_heatmap.jpg`: Visual representation of:
  - GPU-to-GPU bandwidth measurements
  - Communication capabilities between nodes

- `traffic_epoch_N.jpg`: Visual representation of:
  - Traffic patterns between nodes for epoch N
  - Shows the amount of data transferred during reduce-scatter and all-gather
  - Helps identify communication patterns and bottlenecks

## Performance Metrics

The implementation measures and logs:
1. Bandwidth between GPUs (GB/s)
2. Traffic volume during reduce-scatter and all-gather operations (MB) for each epoch
3. Training time per epoch and step
4. Overall training duration
5. Buffer usage and potential overflows 