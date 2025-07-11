#include "Matrix.h"
#include <iostream>

template <typename T, int ROWS, int COLS>
Matrix<T, ROWS, COLS>(T defaultVal = 0) {
    for(int i = 0; i < ROWS*COLS; i++) {
        hostData[i] = defaultVal;
    }
}

template <typename T, int ROWS, int COLS>
tempRow Matrix<T, ROWS, COLS>::operator[](int row) {
    assert(row >= 0 && row < ROWS);
    return tempRow(this, row);
}

template <typename T, int ROWS, int COLS>
Matrix<T, ROWS, COLS> Matrix<T, ROWS, COLS>::operator+(const Matrix<T, ROWS, COLS>& hostB) {
    assert(ROWS == hostB.getRows() && COLS == hostB.getCols);
    Matrix<T, ROWS, COLS> hostC;
    T* devA;
    T* devB;
    T* devC;
    size_t size = ROWS*COLS*sizeof(T);

    cudaError_t error = cudaSuccess;

    error = cudaMalloc((void**)&devA, size);
    if(error != cudaSuccess) {
        std::cerr << "Failed to allocate devA for Matrix Addition" << std::endl;
        return NULL->hostData;
    }
    error = cudaMalloc((void**)&devB, size);
    if(error != cudaSuccess) {
        std::cerr << "Failed to allocate devB for Matrix Addition" << std::endl;
        return NULL;
    }
    error = cudaMalloc((void**)&devC, size);
    if(error != cudaSuccess) {
        std::cerr << "Failed to allocate devC for Matrix Addition" << std::endl;
        return NULL;
    }

    error = cudaMemcpy(devA, this->hostData, size, cudaMemcpyHostToDevice);
    if(error != cudaSuccess) {
        std::cerr << "Failed to copy Matrix A from host to device" << std::endl;
        return NULL;
    }

    error = cudaMemcpy(devB, hostB.hostData, size, cudaMemcpyHostToDevice);
    if(error != cudaSuccess) {
        std::cerr << "Failed to copy Matrix B from host to device" << std::endl;
        return NULL;
    }


    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    return hostC;
}

template<typename T, int ROWS, int COLS>
template<int B_ROWS, int B_COLS>
Matrix<T, ROWS, B_COLS> Matrix<T, ROWS, COLS>::operator*(const Matrix<T, B_ROWS, B_COLS>& hostB) {

}
