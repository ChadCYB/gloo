#include <gloo/transport/tcp/device.h>
#include <gloo/rendezvous/context.h>
#include <gloo/rendezvous/file_store.h>
#include <gloo/transport/tcp/pair.h>
#include <gloo/algorithm.h>
#include <gloo/allreduce_ring.h>

#include <memory>
#include <vector>
#include <thread>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include "alexnet_cuda.h"
#include <fstream>
#include <sstream>
#include <filesystem>
#include "config_parser.h"
#include "topology.h"
#include <nlohmann/json.hpp>
#include <cstdlib>

// Constants
constexpr size_t MAX_BUFFER_SIZE = 1024 * 1024 * 1024;  // 1GB buffer limit

class DistributedTrainer {
public:
    DistributedTrainer(int rank, int size, const std::string& config_path) {
        if (rank < 0 || size <= 0 || rank >= size) {
            throw std::invalid_argument("Invalid rank or size");
        }

        rank_ = rank;
        size_ = size;

        // Parse configuration
        ConfigParser config_parser(config_path);
        training_config_ = config_parser.get_training_config();
        distributed_config_ = config_parser.get_distributed_config();
        logging_config_ = config_parser.get_logging_config();

        // Ensure topology configuration is valid
        if (distributed_config_.topology.num_nodes != size_) {
            throw std::runtime_error("Topology num_nodes does not match process count");
        }

        // Initialize topology
        topology_ = Topology::create(distributed_config_.topology);

        // Initialize model after configuration is parsed
        model_ = std::make_unique<AlexNetCUDA>(rank_);

        // Create Gloo context
        gloo::transport::tcp::attr attr;
        attr.hostname = distributed_config_.use_localhost ? "127.0.0.1" : distributed_config_.host;
        auto dev = gloo::transport::tcp::CreateDevice(attr);

        // Get store path from environment or use default
        std::string store_path = "/tmp/gloo";
        if (const char* env_store_path = std::getenv("STORE_PATH")) {
            store_path = env_store_path;
        }

        // Create store for rendezvous
        auto store = std::make_shared<gloo::rendezvous::FileStore>(store_path);

        // Initialize context
        context_ = std::make_shared<gloo::rendezvous::Context>(rank_, size_);
        context_->setTimeout(std::chrono::seconds(30));
        context_->connectFullMesh(store, dev);

        // Initialize parameter buffer
        std::vector<float> params;
        model_->getParameters(params);
        parameter_size_ = params.size();
        parameter_buffer_.resize(parameter_size_);
        gradient_buffer_.resize(parameter_size_);

        // Initialize traffic matrix for all ranks
        traffic_matrix_.resize(size_, std::vector<size_t>(size_, 0));
        epoch_traffic_.resize(training_config_.num_epochs, 
                            std::vector<std::vector<size_t>>(size_, 
                            std::vector<size_t>(size_, 0)));

        // Create results directory if it doesn't exist
        if (rank_ == 0) {
            std::filesystem::create_directories(logging_config_.output_dir);
        }

        // Open metrics file for rank 0
        if (rank_ == 0) {
            std::string metrics_file_path = logging_config_.output_dir + "/training_metrics.txt";
            metrics_file_.open(metrics_file_path);
            if (!metrics_file_.is_open()) {
                throw std::runtime_error("Failed to open metrics file: " + metrics_file_path);
            }

            std::cout << "Using GPU " << rank_ % getNumGPUs() << " for rank " << rank_ << std::endl;
            std::cout << "Total parameters: " << parameter_size_ << std::endl;
            
            // Initialize metrics file with debug output
            std::cout << "Opening metrics file..." << std::endl;
            metrics_file_ << "Training Metrics Log\n";
            metrics_file_ << "===================\n\n";
            metrics_file_ << "Configuration:\n";
            metrics_file_ << "Number of epochs: " << training_config_.num_epochs << "\n";
            metrics_file_ << "Steps per epoch: " << training_config_.steps_per_epoch << "\n";
            metrics_file_ << "Learning rate: " << training_config_.learning_rate << "\n";
            metrics_file_ << "Number of nodes: " << distributed_config_.num_nodes << "\n";
            metrics_file_ << "Host: " << distributed_config_.host << "\n";
            metrics_file_ << "Port: " << distributed_config_.port << "\n\n";
            metrics_file_.flush();

            bandwidth_matrix_.resize(size_, std::vector<double>(size_, 0.0));

            // Measure bandwidth between nodes
            std::cout << "Measuring bandwidth between nodes..." << std::endl;
            metrics_file_ << "Bandwidth Measurements (GB/s):\n";
            metrics_file_.flush();

            for (int i = 0; i < size_; ++i) {
                for (int j = i + 1; j < size_; ++j) {
                    double bandwidth = model_->measureBandwidthTo(j);
                    bandwidth_matrix_[i][j] = bandwidth;
                    bandwidth_matrix_[j][i] = bandwidth;
                    std::cout << "Bandwidth between GPU " << i << " and GPU " << j 
                             << ": " << std::fixed << std::setprecision(3) 
                             << bandwidth << " GB/s" << std::endl;
                    metrics_file_ << "GPU " << i << " <-> GPU " << j 
                                << ": " << std::fixed << std::setprecision(3) 
                                << bandwidth << " GB/s\n";
                    metrics_file_.flush();
                }
            }
            metrics_file_ << "\nBandwidth Matrix (GB/s):\n";
            for (int i = 0; i < size_; ++i) {
                for (int j = 0; j < size_; ++j) {
                    metrics_file_ << std::fixed << std::setprecision(3) 
                                << bandwidth_matrix_[i][j] << "\t";
                }
                metrics_file_ << "\n";
            }
            metrics_file_ << "\n";
            metrics_file_.flush();
        }

        run_first_batch_ = training_config_.run_first_batch;
        if (run_first_batch_) {
            std::cout << "Running in first batch mode - will only execute first batch" << std::endl;
        }

        // Initialize bandwidth limit from configuration
        bandwidth_limit_ = distributed_config_.bandwidth_limit;
    }

    void train() {
        if (rank_ == 0) {
            std::cout << "Starting training with " << training_config_.num_epochs << " epochs..." << std::endl;
            metrics_file_ << "Training Metrics:\n";
            metrics_file_ << "Number of nodes: " << size_ << "\n";
            metrics_file_ << "Number of epochs: " << training_config_.num_epochs << "\n";
            metrics_file_ << "Steps per epoch: " << training_config_.steps_per_epoch << "\n";
            metrics_file_ << "First batch mode: " << (run_first_batch_ ? "true" : "false") << "\n\n";
            metrics_file_.flush();
        }

        auto total_start = std::chrono::high_resolution_clock::now();
        
        int num_epochs_to_run = run_first_batch_ ? 1 : training_config_.num_epochs;
        epoch_times_.resize(num_epochs_to_run);
        
        for (int epoch = 0; epoch < num_epochs_to_run; ++epoch) {
            auto epoch_start = std::chrono::high_resolution_clock::now();
            
            // Reset traffic for this epoch
            if (rank_ == 0) {
                for (int i = 0; i < size_; ++i) {
                    for (int j = 0; j < size_; ++j) {
                        traffic_matrix_[i][j] = 0;
                    }
                }
            }
            
            for (int step = 0; step < training_config_.steps_per_epoch; ++step) {
                auto step_start = std::chrono::high_resolution_clock::now();
                
                model_->trainStep();
                synchronizeParameters();
                
                auto step_end = std::chrono::high_resolution_clock::now();
                auto step_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    step_end - step_start);
                
                if (rank_ == 0) {
                    std::string step_info = "Epoch " + std::to_string(epoch + 1) + "/" + 
                                          std::to_string(num_epochs_to_run) + ", Step " + 
                                          std::to_string(step + 1) + "/" + 
                                          std::to_string(training_config_.steps_per_epoch) + 
                                          ", Time: " + std::to_string(step_duration.count()) + "ms\n";
                    metrics_file_ << step_info;
                    metrics_file_.flush();
                    std::cout << step_info;
                }
            }
            
            // Save epoch traffic
            if (rank_ == 0) {
                metrics_file_ << "\nEpoch " << (epoch + 1) << " Traffic Matrix (MB):\n";
                std::cout << "\nEpoch " << (epoch + 1) << " Traffic Matrix (MB):\n";
                for (int i = 0; i < size_; ++i) {
                    for (int j = 0; j < size_; ++j) {
                        double traffic_mb = traffic_matrix_[i][j] / (1024.0 * 1024.0);
                        metrics_file_ << std::fixed << std::setprecision(2) << traffic_mb << "\t";
                        std::cout << std::fixed << std::setprecision(2) << traffic_mb << "\t";
                    }
                    metrics_file_ << "\n";
                    std::cout << "\n";
                }
                metrics_file_.flush();
            }
            
            auto epoch_end = std::chrono::high_resolution_clock::now();
            auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(
                epoch_end - epoch_start);
            epoch_times_[epoch] = epoch_duration.count();
            
            if (rank_ == 0) {
                std::string epoch_info = "Epoch " + std::to_string(epoch + 1) + 
                                       " completed in " + std::to_string(epoch_duration.count()) + "s\n";
                metrics_file_ << epoch_info;
                metrics_file_.flush();
                std::cout << epoch_info;
            }
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(
            total_end - total_start);

        if (rank_ == 0) {
            std::string total_info = "\nTotal Training Time: " + 
                                   std::to_string(total_duration.count()) + "s\n";
            metrics_file_ << total_info;
            metrics_file_.flush();
            std::cout << total_info;
            
            // Write epoch times
            metrics_file_ << "\nEpoch Times (seconds):\n";
            for (int i = 0; i < num_epochs_to_run; ++i) {
                metrics_file_ << "Epoch " << (i + 1) << ": " << epoch_times_[i] << "s\n";
            }
            metrics_file_.flush();
        }
    }

    ~DistributedTrainer() {
        if (rank_ == 0) {
            // Write bandwidth matrix at the end
            metrics_file_ << "\nFinal Bandwidth Matrix (GB/s):\n";
            for (int i = 0; i < size_; ++i) {
                for (int j = 0; j < size_; ++j) {
                    metrics_file_ << std::fixed << std::setprecision(3) 
                                << bandwidth_matrix_[i][j] << "\t";
                }
                metrics_file_ << "\n";
            }
            metrics_file_.flush();

            // Write traffic matrix
            metrics_file_ << "\nTraffic Matrix (MB):\n";
            for (int i = 0; i < size_; ++i) {
                for (int j = 0; j < size_; ++j) {
                    metrics_file_ << std::fixed << std::setprecision(2) 
                                << (traffic_matrix_[i][j] / (1024.0 * 1024.0)) << "\t";
                }
                metrics_file_ << "\n";
            }
            metrics_file_.flush();

            if (metrics_file_.is_open()) {
                metrics_file_.close();
            }
        }
    }

private:
    void synchronizeParameters() {
        // Calculate data size per ring based on the number of rings
        size_t num_rings = topology_->getNumRings();
        size_t data_size_per_ring = parameter_size_ * sizeof(float);
        if (num_rings > 0) {
            data_size_per_ring /= num_rings;  // Only divide by number of rings, not 2*rings
        }
        
        // Get reduce-scatter connections for this rank
        auto connections = topology_->getReduceScatterConnections(rank_);
        
        // Update data size for each connection
        for (auto& conn : connections) {
            conn.data_size = data_size_per_ring;
        }
        
        // Simulate communication with bandwidth limits
        for (const auto& conn : connections) {
            double transfer_time = static_cast<double>(conn.data_size) / (bandwidth_limit_ * 1e9);
            traffic_matrix_[conn.src][conn.dst] += conn.data_size;
            
            // Add error checking for buffer overflows
            if (traffic_matrix_[conn.src][conn.dst] > MAX_BUFFER_SIZE) {
                std::cerr << "Warning: Buffer overflow detected between ranks " 
                          << conn.src << " and " << conn.dst << std::endl;
            }
        }
        
        // Get all-gather connections and repeat the process
        connections = topology_->getAllGatherConnections(rank_);
        for (auto& conn : connections) {
            conn.data_size = data_size_per_ring;
        }
        
        for (const auto& conn : connections) {
            double transfer_time = static_cast<double>(conn.data_size) / (bandwidth_limit_ * 1e9);
            traffic_matrix_[conn.src][conn.dst] += conn.data_size;
            
            if (traffic_matrix_[conn.src][conn.dst] > MAX_BUFFER_SIZE) {
                std::cerr << "Warning: Buffer overflow detected between ranks "
                          << conn.src << " and " << conn.dst << std::endl;
            }
        }
    }

    static int getNumGPUs() {
        int num_gpus;
        cudaGetDeviceCount(&num_gpus);
        return num_gpus;
    }

    int rank_;
    int size_;
    std::unique_ptr<AlexNetCUDA> model_;
    std::shared_ptr<gloo::rendezvous::Context> context_;
    size_t parameter_size_;
    std::vector<float> parameter_buffer_;
    std::vector<float> gradient_buffer_;
    std::ofstream metrics_file_;
    std::vector<std::vector<double>> bandwidth_matrix_;
    std::vector<std::vector<size_t>> traffic_matrix_;
    std::vector<std::vector<std::vector<size_t>>> epoch_traffic_;
    ConfigParser::TrainingConfig training_config_;
    ConfigParser::DistributedConfig distributed_config_;
    ConfigParser::LoggingConfig logging_config_;
    float bandwidth_limit_;  // Bandwidth limit in GB/s
    bool run_first_batch_;
    std::vector<double> epoch_times_;
    std::unique_ptr<Topology> topology_;
};

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <rank> <size> <config_path>" << std::endl;
        return 1;
    }

    int rank = std::stoi(argv[1]);
    int size = std::stoi(argv[2]);
    std::string config_path = argv[3];

    try {
        DistributedTrainer trainer(rank, size, config_path);
        trainer.train();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
} 