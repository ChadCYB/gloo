#pragma once

#include <string>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

class ConfigParser {
public:
    struct TrainingConfig {
        int num_epochs;
        int steps_per_epoch;
        float learning_rate;
        bool run_first_batch;
    };

    struct TopologyConfig {
        std::string type;           // "ring", "mesh", "hierarchical"
        int num_nodes;              // Number of nodes in the topology
        std::vector<int> permutations; // Permutations for ring topology
    };

    struct DistributedConfig {
        int num_nodes;
        bool use_localhost;
        std::string host;
        int port;
        float bandwidth_limit;      // Bandwidth limit in GB/s
        TopologyConfig topology;    // Topology configuration
    };

    struct LoggingConfig {
        std::string level;
        std::string output_dir;
    };

    ConfigParser(const std::string& config_path) {
        std::ifstream file(config_path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open config file: " + config_path);
        }

        try {
            json_ = nlohmann::json::parse(file);
            validate_config();

            // Read topology config if specified
            if (json_["distributed"].contains("topology_config_file")) {
                std::string topology_config_path = json_["distributed"]["topology_config_file"];
                std::ifstream topology_file(topology_config_path);
                if (!topology_file.is_open()) {
                    throw std::runtime_error("Failed to open topology config file: " + topology_config_path);
                }
                topology_json_ = nlohmann::json::parse(topology_file);
                validate_topology_config();
            }
        } catch (const nlohmann::json::parse_error& e) {
            throw std::runtime_error("Failed to parse config file: " + std::string(e.what()));
        }
    }

    TrainingConfig get_training_config() const {
        TrainingConfig config;
        config.num_epochs = json_["training"]["num_epochs"];
        config.steps_per_epoch = json_["training"]["steps_per_epoch"];
        config.learning_rate = json_["training"]["learning_rate"];
        config.run_first_batch = json_["training"]["run_first_batch"];
        return config;
    }

    DistributedConfig get_distributed_config() const {
        DistributedConfig config;
        config.use_localhost = json_["distributed"]["use_localhost"];
        config.host = json_["distributed"]["host"];
        config.port = json_["distributed"]["port"];
        config.bandwidth_limit = json_["distributed"]["bandwidth_limit"];
        
        // Parse topology configuration from either main config or topology config file
        if (json_["distributed"].contains("topology_config_file")) {
            config.topology.num_nodes = topology_json_["num_nodes"];
            config.topology.type = topology_json_["topology_type"];
            config.topology.permutations = topology_json_["permutations"].get<std::vector<int>>();
        } else {
            config.num_nodes = json_["distributed"]["num_nodes"];
            config.topology.type = json_["distributed"]["topology"]["type"];
            config.topology.num_nodes = config.num_nodes;
            config.topology.permutations = {0}; // Default to single ring
        }
        
        return config;
    }

    LoggingConfig get_logging_config() const {
        LoggingConfig config;
        config.level = json_["logging"]["level"];
        config.output_dir = json_["logging"]["output_dir"];
        return config;
    }

private:
    void validate_config() {
        // Validate training config
        if (!json_.contains("training")) throw std::runtime_error("Missing training config");
        if (!json_["training"].contains("num_epochs")) throw std::runtime_error("Missing num_epochs");
        if (!json_["training"].contains("steps_per_epoch")) throw std::runtime_error("Missing steps_per_epoch");
        if (!json_["training"].contains("learning_rate")) throw std::runtime_error("Missing learning_rate");
        if (!json_["training"].contains("run_first_batch")) throw std::runtime_error("Missing run_first_batch");

        // Validate distributed config
        if (!json_.contains("distributed")) throw std::runtime_error("Missing distributed config");
        if (!json_["distributed"].contains("use_localhost")) throw std::runtime_error("Missing use_localhost");
        if (!json_["distributed"].contains("host")) throw std::runtime_error("Missing host");
        if (!json_["distributed"].contains("port")) throw std::runtime_error("Missing port");
        if (!json_["distributed"].contains("bandwidth_limit")) throw std::runtime_error("Missing bandwidth_limit");

        // Validate logging config
        if (!json_.contains("logging")) throw std::runtime_error("Missing logging config");
        if (!json_["logging"].contains("level")) throw std::runtime_error("Missing logging level");
        if (!json_["logging"].contains("output_dir")) throw std::runtime_error("Missing output_dir");
    }

    void validate_topology_config() {
        if (!topology_json_.contains("num_nodes")) throw std::runtime_error("Missing num_nodes in topology config");
        if (!topology_json_.contains("topology_type")) throw std::runtime_error("Missing topology_type in topology config");
        if (!topology_json_.contains("permutations")) throw std::runtime_error("Missing permutations in topology config");
    }

    nlohmann::json json_;
    nlohmann::json topology_json_;
}; 