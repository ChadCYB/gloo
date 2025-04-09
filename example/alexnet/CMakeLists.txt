cmake_minimum_required(VERSION 3.10)
project(alexnet_trainer)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDA REQUIRED)

# Add Gloo subdirectory
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../ ${CMAKE_BINARY_DIR}/gloo)

# Add nlohmann_json
include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(json)

# Add executable
add_executable(alexnet_trainer alexnet_trainer.cpp)

# Include directories
target_include_directories(alexnet_trainer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(alexnet_trainer PRIVATE
    gloo
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    nlohmann_json::nlohmann_json
    stdc++fs
)

# Set CUDA architecture
set_target_properties(alexnet_trainer PROPERTIES
    CUDA_ARCHITECTURES "75"
) 