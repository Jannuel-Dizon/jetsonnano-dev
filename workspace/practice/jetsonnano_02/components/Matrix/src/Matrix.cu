#include "Matrix.h"
#include <iostream>

template <typename T, int A_ROWS, int A_COLS>
MyLA::Matrix<T, ROWS, COLS>(T defaultVal) {
    for(int i = 0; i < A_ROWS*A_COLS; i++) {
        hostData[i] = defaultVal;
    }
}

// ROWS = y
// COLS = x
template <typename T, int A_ROWS, int A_COLS>
template <int B_ROWS, int B_COLS>
MyLA::Matrix<T, A_ROWS, A_COLS> MyLA::Matrix<T, A_ROWS, A_COLS>::operator+(const Matrix<T, B_ROWS, B_COLS>& hostB) const {
    assert(A_ROWS == B_ROWS && A_COLS == B_COLS);
    MyLA::Matrix<T, A_ROWS, A_COLS> hostC;
    T* devA;
    T* devB;
    T* devC;
    size_t size = A_ROWS*A_COLS*sizeof(T);

    cudaError_t error = cudaSuccess;

    error = cudaMalloc((void**)&devA, size);
    if(error != cudaSuccess) {
        std::cerr << "Failed to allocate devA for Matrix Addition" << std::endl;
        throw std::runtime_error("Failed to allocate device memory.");
    }
    error = cudaMalloc((void**)&devB, size);
    if(error != cudaSuccess) {
        std::cerr << "Failed to allocate devB for Matrix Addition" << std::endl;
        throw std::runtime_error("Failed to allocate device memory.");
    }
    error = cudaMalloc((void**)&devC, size);
    if(error != cudaSuccess) {
        std::cerr << "Failed to allocate devC for Matrix Addition" << std::endl;
        throw std::runtime_error("Failed to allocate device memory.");
    }

    error = cudaMemcpy(devA, this->getData(), size, cudaMemcpyHostToDevice);
    if(error != cudaSuccess) {
        std::cerr << "Failed to copy Matrix A from host to device" << std::endl;
        throw std::runtime_error("Failed to copy host to device memory.");
    }

    error = cudaMemcpy(devB, hostB.getData(), size, cudaMemcpyHostToDevice);
    if(error != cudaSuccess) {
        std::cerr << "Failed to copy Matrix B from host to device" << std::endl;
        throw std::runtime_error("Failed to copy host to device memory.");
    }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((A_COLS+threadsPerBlock.x-1)/threadsPerBlock.x, (A_ROWS+threadsPerBlock.y-1)/threadsPerBlock.y);

    matAddKernel<<<numBlocks, threadsPerBlock>>>(devA, devB, devC, A_ROWS, A_COLS);

    error = cudaMemcpy(hostC.getData(), devC, size, cudaMemcpyDeviceToHost);
    if(error != cudaSuccess) {
        std::cerr << "Failed to copy Matrix C from device to host" << std::endl;
        throw std::runtime_error("Failed to copy device to host memory.");
    }

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    return hostC;
}

template <typename T>
__global__ void matAddKernel(T* devA, T* devB, T* devC, int rows, int cols) {
    int row = (blockIdx.y * blockDim.y) + threadIdx.y;
    int col = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(row < rows && col < cols) {
        int index = (row*cols) + col;
        devC[index] = devA[index] + devB[index];
    }
}

template<typename T, int A_ROWS, int A_COLS>
template<int B_ROWS, int B_COLS>
MyLA::Matrix<T, A_ROWS, B_COLS> MyLA::Matrix<T, A_ROWS, A_COLS>::operator*(const Matrix<T, B_ROWS, B_COLS>& hostB) const {
    assert(A_COLS == B_COLS);
    MyLA::Matrix<T, A_ROWS, B_COLS> hostC;
    T* devA;
    T* devB;
    T* devC;
    size_t sizeA = A_ROWS*A_COLS*sizeof(T);
    size_t sizeB = B_ROWS*B_COLS*sizeof(T);
    size_t sizeC = A_ROWS*B_COLS*sizeof(T);
    cudaError_t error;

    error = cudaMalloc((void**)&devA, sizeA);
    if(error != cudaSuccess) {
        std::cerr << "Failed to allocate devA for Matrix Matrix Multiplication" << std::endl;
        throw std::runtime_error("Failed to allocate device memory.");
    }

    error = cudaMalloc((void**)&devB, sizeB);
    if(error != cudaSuccess) {
        std::cerr << "Failed to allocate devB for Matrix Matrix Multiplication" << std::endl;
        throw std::runtime_error("Failed to allocate device memory.");
    }

    error = cudaMalloc((void**)&devC, sizeC);
    if(error != cudaSuccess) {
        std::cerr << "Failed to allocate devC for Matrix Matrix Multiplication" << std::endl;
        throw std::runtime_error("Failed to allocate device memory.");
    }

    error = cudaMemcpy(devA, this->getData(), sizeA, cudaMemcpyHostToDevice);
    if(error != cudaSuccess) {
        std::cerr << "Failed to copy Matrix A from host to device" << std::endl;
        throw std::runtime_error("Failed to copy host to device memory.");
    }

    error = cudaMemcpy(devB, hostB.getData(), sizeB, cudaMemcpyHostToDevice);
    if(error != cudaSuccess) {
        std::cerr << "Failed to copy Matrix B from host to device" << std::endl;
        throw std::runtime_error("Failed to copy host to device memory.");
    }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((B_COLS+threadsPerBlock.x-1)/threadsPerBlock.x, (A_ROWS+threadsPerBlock.y-1)/threadsPerBlock.y);

    matmatMulKernel<<<numBlocks, threadsPerBlock>>>(devA, devB, devC, A_ROWS, A_COLS, B_COLS);

    error = cudaMemcpy(hostC.getData(), devC, sizeC, cudaMemcpyDeviceToHost);
    if(error != cudaSuccess) {
        std::cerr << "Failed to copy Matrix C from device to host" << std::endl;
        throw std::runtime_error("Failed to copy device to host memory.");
    }

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    return hostC;
}

template<typename T>
__global__ void matmatMulKernel(T* devA, T* devB, T* devC, int rowsA, int colsrowsAB, int colsB) {
    int row = (blockIdx.y*blockDim.y) + threadIdx.y;
    int col = (blockIdx.x*blockDim.x) + threadIdx.x;
    if(row < rowsA && col < colsB) {
        T accum = 0;
        for(int i = 0; i < colsrowsAB; i++) {
            accum += devA[(row*colsrowsAB)+i] * devB[(i*colsB)+col];
        }
        int indexC = (row*colsB) + col;
        devC[indexC] = accum;
    }
}
