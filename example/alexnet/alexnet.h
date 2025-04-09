#pragma once

#include <vector>
#include <memory>
#include <random>
#include <cmath>

// Simple AlexNet-like architecture for demonstration
class AlexNet {
public:
    struct Layer {
        std::vector<float> weights;
        std::vector<float> biases;
        std::vector<float> gradients;
        std::vector<float> bias_gradients;
        int in_channels;
        int out_channels;
        int kernel_size;
        
        Layer(int in_c, int out_c, int k_size) 
            : in_channels(in_c), out_channels(out_c), kernel_size(k_size) {
            int weight_size = in_channels * out_channels * kernel_size * kernel_size;
            weights.resize(weight_size);
            gradients.resize(weight_size);
            biases.resize(out_channels);
            bias_gradients.resize(out_channels);
            
            // Initialize weights with Xavier initialization
            float scale = std::sqrt(2.0f / (in_channels * kernel_size * kernel_size));
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> d(0, scale);
            
            for (auto& w : weights) {
                w = d(gen);
            }
            for (auto& b : biases) {
                b = 0.0f;
            }
        }
    };

    AlexNet(int batch_size = 32) : batch_size_(batch_size) {
        // Simplified AlexNet architecture
        layers_.emplace_back(3, 64, 11);    // Conv1
        layers_.emplace_back(64, 192, 5);   // Conv2
        layers_.emplace_back(192, 384, 3);  // Conv3
        layers_.emplace_back(384, 256, 3);  // Conv4
        layers_.emplace_back(256, 256, 3);  // Conv5
        layers_.emplace_back(256, 4096, 1); // FC6
        layers_.emplace_back(4096, 4096, 1);// FC7
        layers_.emplace_back(4096, 1000, 1);// FC8 (1000 classes)
    }

    // Get total number of parameters for gradient synchronization
    size_t getTotalParameters() const {
        size_t total = 0;
        for (const auto& layer : layers_) {
            total += layer.weights.size() + layer.biases.size();
        }
        return total;
    }

    // Get all parameters as a flat vector for synchronization
    std::vector<float> getParameters() const {
        std::vector<float> params;
        size_t total_size = getTotalParameters();
        params.reserve(total_size);
        
        for (const auto& layer : layers_) {
            params.insert(params.end(), layer.weights.begin(), layer.weights.end());
            params.insert(params.end(), layer.biases.begin(), layer.biases.end());
        }
        return params;
    }

    // Get all gradients as a flat vector for synchronization
    std::vector<float> getGradients() const {
        std::vector<float> grads;
        size_t total_size = getTotalParameters();
        grads.reserve(total_size);
        
        for (const auto& layer : layers_) {
            grads.insert(grads.end(), layer.gradients.begin(), layer.gradients.end());
            grads.insert(grads.end(), layer.bias_gradients.begin(), layer.bias_gradients.end());
        }
        return grads;
    }

    // Set parameters from a flat vector after synchronization
    void setParameters(const std::vector<float>& params) {
        size_t offset = 0;
        for (auto& layer : layers_) {
            std::copy(params.begin() + offset, 
                     params.begin() + offset + layer.weights.size(),
                     layer.weights.begin());
            offset += layer.weights.size();
            
            std::copy(params.begin() + offset,
                     params.begin() + offset + layer.biases.size(),
                     layer.biases.begin());
            offset += layer.biases.size();
        }
    }

    // Set gradients from a flat vector after synchronization
    void setGradients(const std::vector<float>& grads) {
        size_t offset = 0;
        for (auto& layer : layers_) {
            std::copy(grads.begin() + offset,
                     grads.begin() + offset + layer.gradients.size(),
                     layer.gradients.begin());
            offset += layer.gradients.size();
            
            std::copy(grads.begin() + offset,
                     grads.begin() + offset + layer.bias_gradients.size(),
                     layer.bias_gradients.begin());
            offset += layer.bias_gradients.size();
        }
    }

    // Simulate one training step (just accumulate some fake gradients for demo)
    void trainStep() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> d(0, 0.01);
        
        for (auto& layer : layers_) {
            for (auto& grad : layer.gradients) {
                grad = d(gen);
            }
            for (auto& grad : layer.bias_gradients) {
                grad = d(gen);
            }
        }
    }

    // Apply gradients with given learning rate
    void applyGradients(float learning_rate) {
        for (auto& layer : layers_) {
            for (size_t i = 0; i < layer.weights.size(); ++i) {
                layer.weights[i] -= learning_rate * layer.gradients[i];
            }
            for (size_t i = 0; i < layer.biases.size(); ++i) {
                layer.biases[i] -= learning_rate * layer.bias_gradients[i];
            }
        }
    }

private:
    std::vector<Layer> layers_;
    int batch_size_;
}; 