#pragma once

#include <vector>
#include <map>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include "config_parser.h"

class Topology {
public:
    struct Connection {
        int src;
        int dst;
        size_t data_size;
        int ring_id;  // Added to identify which ring the connection belongs to
    };

    static std::unique_ptr<Topology> create(const ConfigParser::TopologyConfig& config) {
        return std::make_unique<Topology>(config);
    }

    Topology(const ConfigParser::TopologyConfig& config) 
        : config_(config) {
        initializeConnections();
    }

    std::vector<Connection> getReduceScatterConnections(int rank) const {
        std::vector<Connection> connections;
        
        if (config_.type == "ring") {
            // Calculate data size per ring
            size_t data_size_per_ring = 0;  // Will be set by the caller
            
            // For each permutation in the configuration
            for (size_t ring_id = 0; ring_id < config_.permutations.size(); ++ring_id) {
                // Create a permutation of nodes based on the configuration
                std::vector<int> permutation(config_.num_nodes);
                for (int i = 0; i < config_.num_nodes; ++i) {
                    permutation[i] = i;
                }
                
                // Apply the rotation specified in the permutation
                int rotation = config_.permutations[ring_id];
                std::rotate(permutation.begin(), 
                          permutation.begin() + rotation,
                          permutation.end());
                
                // Find the current node's position in the permutation
                int pos = std::find(permutation.begin(), permutation.end(), rank) - permutation.begin();
                
                // Find the next node in this ring's permutation
                int next_pos = (pos + 1) % config_.num_nodes;
                int next = permutation[next_pos];
                
                // Only add forward connection - backward connection is handled by the other node
                connections.push_back({rank, next, data_size_per_ring, static_cast<int>(ring_id)});
            }
        }
        else if (config_.type == "hierarchical") {
            // Hierarchical ring topology
            int level_size = config_.num_nodes / 2; // Assuming 2 levels for now
            int level = rank / level_size;
            int pos_in_level = rank % level_size;
            
            // Connect within level
            int next_in_level = level * level_size + ((pos_in_level + 1) % level_size);
            connections.push_back({rank, next_in_level, 0, 0});
            
            // Connect between levels
            if (level < 1) { // Assuming 2 levels
                int next_level_node = (level + 1) * level_size + pos_in_level;
                connections.push_back({rank, next_level_node, 0, 1});
            }
        }
        
        return connections;
    }

    std::vector<Connection> getAllGatherConnections(int rank) const {
        // For most topologies, all-gather uses the same connection pattern as reduce-scatter
        return getReduceScatterConnections(rank);
    }

    void updateTrafficMatrix(std::vector<std::vector<size_t>>& traffic_matrix, 
                           const std::vector<Connection>& connections,
                           size_t data_size) const {
        for (const auto& conn : connections) {
            traffic_matrix[conn.src][conn.dst] += data_size;
            traffic_matrix[conn.dst][conn.src] += data_size;
        }
    }

    size_t getNumRings() const { return num_rings_; }

private:
    void initializeConnections() {
        // Validate topology configuration
        if (config_.type != "ring" && config_.type != "hierarchical") {
            throw std::runtime_error("Unsupported topology type: " + config_.type);
        }
        
        if (config_.num_nodes <= 0) {
            throw std::runtime_error("Number of nodes must be positive");
        }
        
        if (config_.permutations.empty()) {
            throw std::runtime_error("At least one permutation must be specified");
        }
        
        for (int rotation : config_.permutations) {
            if (rotation < 0 || rotation >= config_.num_nodes) {
                throw std::runtime_error("Permutation rotations must be between 0 and num_nodes-1");
            }
        }

        if (config_.type == "ring") {
            num_rings_ = config_.permutations.empty() ? 1 : config_.permutations.size();
        } else if (config_.type == "hierarchical") {
            num_rings_ = 2;  // Two levels in hierarchical topology
        }
    }

    ConfigParser::TopologyConfig config_;
    size_t num_rings_;
}; 