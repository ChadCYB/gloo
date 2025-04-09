#ifndef ALEXNET_CUDA_H
#define ALEXNET_CUDA_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "CUBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

class AlexNetCUDA {
private:
    int rank_;
    cublasHandle_t cublas_handle_;
    float* d_parameter_buffer_;
    float* d_gradient_buffer_;
    size_t total_parameters_;
    std::vector<size_t> layer_sizes_;
    std::vector<float*> d_weights_;
    std::vector<float*> d_biases_;
    std::vector<float*> d_gradients_;
    std::vector<float*> d_bias_gradients_;
    std::vector<size_t> weight_sizes_;
    std::vector<size_t> bias_sizes_;

    static int getNumGPUs() {
        int count;
        CUDA_CHECK(cudaGetDeviceCount(&count));
        return count;
    }

    // Measure bandwidth between two GPUs
    double measureBandwidth(int src_rank, int dst_rank, size_t size) {
        if (rank_ != src_rank && rank_ != dst_rank) return 0.0;

        float* src_buffer;
        float* dst_buffer;
        CUDA_CHECK(cudaMalloc(&src_buffer, size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dst_buffer, size * sizeof(float)));

        // Initialize source buffer
        std::vector<float> host_buffer(size, 1.0f);
        CUDA_CHECK(cudaMemcpy(src_buffer, host_buffer.data(), 
                            size * sizeof(float), cudaMemcpyHostToDevice));

        // Create CUDA events for timing
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Warm up (perform multiple transfers)
        if (rank_ == src_rank) {
            for (int i = 0; i < 10; ++i) {
                CUDA_CHECK(cudaMemcpy(dst_buffer, src_buffer, 
                                    size * sizeof(float), cudaMemcpyDeviceToDevice));
            }
        }

        // Measure bandwidth with multiple transfers for better accuracy
        const int num_transfers = 100;
        CUDA_CHECK(cudaEventRecord(start));
        if (rank_ == src_rank) {
            for (int i = 0; i < num_transfers; ++i) {
                CUDA_CHECK(cudaMemcpy(dst_buffer, src_buffer, 
                                    size * sizeof(float), cudaMemcpyDeviceToDevice));
            }
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        // Cleanup
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(src_buffer));
        CUDA_CHECK(cudaFree(dst_buffer));

        // Convert to GB/s (total data transferred / time)
        double total_bytes = size * sizeof(float) * num_transfers;
        double bandwidth = (total_bytes / (milliseconds * 1e-3)) / (1024.0 * 1024.0 * 1024.0);
        return bandwidth;
    }

public:
    AlexNetCUDA(int rank) : rank_(rank) {
        // Initialize CUDA
        CUDA_CHECK(cudaSetDevice(rank_ % getNumGPUs()));
        CUBLAS_CHECK(cublasCreate(&cublas_handle_));
        
        // Simplified AlexNet architecture (much smaller model)
        layer_sizes_ = {3*16*16, 32*8*8, 64*4*4, 128, 10};
        weight_sizes_ = {3*32*3*3, 32*64*3*3, 64*128*4*4, 128*10};
        bias_sizes_ = {32, 64, 128, 10};
        
        // Calculate total parameters
        total_parameters_ = 0;
        for (size_t i = 0; i < weight_sizes_.size(); ++i) {
            total_parameters_ += weight_sizes_[i] + bias_sizes_[i];
        }
        
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_parameter_buffer_, total_parameters_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gradient_buffer_, total_parameters_ * sizeof(float)));
        
        // Initialize weights and biases
        float* current_param = d_parameter_buffer_;
        float* current_grad = d_gradient_buffer_;
        
        for (size_t i = 0; i < weight_sizes_.size(); ++i) {
            d_weights_.push_back(current_param);
            d_gradients_.push_back(current_grad);
            current_param += weight_sizes_[i];
            current_grad += weight_sizes_[i];
            
            d_biases_.push_back(current_param);
            d_bias_gradients_.push_back(current_grad);
            current_param += bias_sizes_[i];
            current_grad += bias_sizes_[i];
            
            // Initialize weights with Xavier initialization
            std::vector<float> host_weights(weight_sizes_[i]);
            std::vector<float> host_biases(bias_sizes_[i]);
            
            std::random_device rd;
            std::mt19937 gen(rd());
            float scale = std::sqrt(2.0f / (layer_sizes_[i] + layer_sizes_[i+1]));
            std::normal_distribution<float> dist(0.0f, scale);
            
            for (size_t j = 0; j < weight_sizes_[i]; ++j) {
                host_weights[j] = dist(gen);
            }
            for (size_t j = 0; j < bias_sizes_[i]; ++j) {
                host_biases[j] = 0.0f;
            }
            
            CUDA_CHECK(cudaMemcpy(d_weights_[i], host_weights.data(), 
                                weight_sizes_[i] * sizeof(float), 
                                cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_biases_[i], host_biases.data(), 
                                bias_sizes_[i] * sizeof(float), 
                                cudaMemcpyHostToDevice));
        }
        
        if (rank_ == 0) {
            std::cout << "Total parameters: " << total_parameters_ << std::endl;
        }
    }

    ~AlexNetCUDA() {
        // Cleanup CUDA resources
        CUBLAS_CHECK(cublasDestroy(cublas_handle_));
        CUDA_CHECK(cudaFree(d_parameter_buffer_));
        CUDA_CHECK(cudaFree(d_gradient_buffer_));
    }

    void getParameters(std::vector<float>& params) {
        params.resize(total_parameters_);
        CUDA_CHECK(cudaMemcpy(params.data(), d_parameter_buffer_, 
                            total_parameters_ * sizeof(float), 
                            cudaMemcpyDeviceToHost));
    }

    void setParameters(const std::vector<float>& params) {
        CUDA_CHECK(cudaMemcpy(d_parameter_buffer_, params.data(), 
                            total_parameters_ * sizeof(float), 
                            cudaMemcpyHostToDevice));
    }

    void getGradients(std::vector<float>& grads) {
        grads.resize(total_parameters_);
        CUDA_CHECK(cudaMemcpy(grads.data(), d_gradient_buffer_, 
                            total_parameters_ * sizeof(float), 
                            cudaMemcpyDeviceToHost));
    }

    void setGradients(const std::vector<float>& grads) {
        CUDA_CHECK(cudaMemcpy(d_gradient_buffer_, grads.data(), 
                            total_parameters_ * sizeof(float), 
                            cudaMemcpyHostToDevice));
    }

    void trainStep() {
        // Simulate forward and backward pass
        std::vector<float> host_gradients(total_parameters_);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.01f);
        
        for (size_t i = 0; i < total_parameters_; ++i) {
            host_gradients[i] = dist(gen);
        }
        
        setGradients(host_gradients);
        
        // Apply gradients with learning rate
        float learning_rate = 0.01f;
        float alpha = -learning_rate;
        
        for (size_t i = 0; i < weight_sizes_.size(); ++i) {
            CUBLAS_CHECK(cublasSaxpy(cublas_handle_, weight_sizes_[i],
                        &alpha, d_gradients_[i], 1,
                        d_weights_[i], 1));
            
            CUBLAS_CHECK(cublasSaxpy(cublas_handle_, bias_sizes_[i],
                        &alpha, d_bias_gradients_[i], 1,
                        d_biases_[i], 1));
        }
    }

    double measureBandwidthTo(int other_rank) {
        // Measure bandwidth between current rank and other rank
        const size_t test_size = 1024 * 1024;  // 1M elements
        double bandwidth = measureBandwidth(rank_, other_rank, test_size);
        if (bandwidth == 0.0) {
            bandwidth = measureBandwidth(other_rank, rank_, test_size);
        }
        return bandwidth;
    }
};

#endif // ALEXNET_CUDA_H 