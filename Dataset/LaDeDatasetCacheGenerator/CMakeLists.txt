cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(generate_cache)

set(Torch_DIR /home/jimmy/anaconda3/envs/Py311/lib/python3.11/site-packages/torch/share/cmake/Torch/)

find_package(CUDAToolkit REQUIRED)
set(CUDA_NVCC_EXECUTABLE "/usr/lib/nvidia-cuda-toolkit/bin/nvcc")

# Here I did following change:
# Changed caffe2 code
# CMake Error at /home/jimmy/anaconda3/envs/Py311/lib/python3.11/site-packages/torch/share/cmake/Caffe2/public/cuda.cmake:78 (message):
# PyTorch requires CUDA 11.0 or above.

message("nvcc:" ${CUDA_NVCC_EXECUTABLE}$)
message("cuda include dir" ${CUDA_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR})

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message(STATUS "CXX Flags: " ${CMAKE_CXX_FLAGS})
message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")
message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}")

find_package(Matplot++ REQUIRED)

find_package(OpenMP REQUIRED)

set(SOURCES
    SegmentGraph.cpp
    LaDeDataset.cpp
    main.cpp
)

add_executable(generate_cache ${SOURCES})
target_link_libraries(generate_cache "${TORCH_LIBRARIES}" Matplot++::matplot OpenMP::OpenMP_CXX)
set_property(TARGET generate_cache PROPERTY CXX_STANDARD 17)

add_definitions(-DOMPI_SKIP_MPICXX)