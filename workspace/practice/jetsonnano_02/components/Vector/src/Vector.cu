#include "Vector.h"
#include "Matrix.h"
#include <iostream>
#include <stdexcept>

#define BLOCKSIZE 256

namespace MyLA {
    namespace Static {

        template<typename T, int SIZE>
        Vector<T, SIZE>(T defaultVal) {
            for(int i = 0; i < SIZE; i++) {
                hostData[i] = defaultVal;
            }
        }

        template<typename T, int SIZE>
        template<int B_SIZE>
        Vector<T, SIZE> Vector<T, SIZE>::operator+(const Vector<T, B_SIZE>& hostB) const {
            if(SIZE != B_SIZE) {
                std::cerr << "Incompatible Vector sizes" << std::endl;
                throw std::runtime_error("Vectors has different sizes");
            }

            Vector<T, SIZE> hostC;
            T* devA;
            T* devB;
            T* devC;
            size_t size = SIZE*sizeof(T);
            cudaError_t error;

            error = cudaMalloc((void**)&devA, size);
            if(error != cudaSuccess) {
                std::cerr << "Failed to allocate devA for Vector Addition" << std::endl;
                throw std::runtime_error("Failed to allocate device memory.");
            }

            error = cudaMalloc((void**)&devB, size);
            if(error != cudaSuccess) {
                std::cerr << "Failed to allocate devA for Vector Addition" << std::endl;
                throw std::runtime_error("Failed to allocate device memory.");
            }

            error = cudaMalloc((void**)&devC, size);
            if(error != cudaSuccess) {
                std::cerr << "Failed to allocate devA for Vector Addition" << std::endl;
                throw std::runtime_error("Failed to allocate device memory.");
            }

            error = cudaMemcpy(devA, this->getData(), size, cudaMemcpyHostToDevice);
            if(error != cudaSuccess) {
                std::cerr << "Failed to copy Vector A from host to device" << std::endl;
                throw std::runtime_error("Failed to copy host to device memory.");
            }

            error = cudaMemcpy(devB, hostB.getData(), size, cudaMemcpyHostToDevice);
            if(error != cudaSuccess) {
                std::cerr << "Failed to copy Vector B from host to device" << std::endl;
                throw std::runtime_error("Failed to copy host to device memory.");
            }

            dim3 threadsPerBlock(BLOCKSIZE);
            dim3 numBlocks((SIZE+BLOCKSIZE-1)/BLOCKSIZE);

            cudaVectorSum<<<numBlocks, threadsPerBlock>>>(devA, devB, devC, SIZE);

            error = cudaMemcpy(hostC.getData(), devC, size, cudaMemcpyDeviceToHost);
            if(error != cudaSuccess) {
                std::cerr << "Failed to copy Vector C from device to host" << std::endl;
                throw std::runtime_error("Failed to copy device to host memory.");
            }

            cudaFree(devA);
            cudaFree(devB);
            cudaFree(devC);

            return hostC;
        }

        template<template T>
        __global__ void cudaVectorSum(const T* devA, const T* devB, T* devC, int size) {
            int index = (blockIdx.x*blockDim.x) + threadIdx.x;
            if(index < size) {
                devC[index] = devA[index] + devB[index];
            }
        }

        template<typename T, int SIZE>
        template<int B_SIZE>
        T Vector<T, SIZE>::dotProduct(const Vector<T, B_SIZE>& hostB) const {
            if(SIZE != B_SIZE) {
                std::cerr << "Incompatible Vector sizes" << std::endl;
                throw std::runtime_error("Vectors has different sizes");
            }

            T result = 0;
            T* devA;
            T* devB;
            size_t size = SIZE*sizeof(T);
            cudaError_t error;

            error = cudaMalloc((void**)&devA, size);
            if(error != cudaSuccess) {
                std::cerr << "Failed to allocate devA for Vector dot product" << std::endl;
                throw std::runtime_error("Failed to allocate device memory.");
            }

            error = cudaMalloc((void**)&devB, size);
            if(error != cudaSuccess) {
                std::cerr << "Failed to allocate devB for Vector dot product" << std::endl;
                throw std::runtime_error("Failed to allocate device memory.");
            }

            error = cudaMemcpy(devA, this->getData(), size, cudaMemcpyHostToDevice);
            if(error != cudaSuccess) {
                std::cerr << "Failed to copy Vector A from host to device" << std::endl;
                throw std::runtime_error("Failed to copy host to device memory.");
            }

            error = cudaMemcpy(devB, hostB.getData(), size, cudaMemcpyHostToDevice);
            if(error != cudaSuccess) {
                std::cerr << "Failed to copy Vector B from host to device" << std::endl;
                throw std::runtime_error("Failed to copy host to device memory.");
            }

            dim3 threadsPerBlock(BLOCKSIZE);
            dim3 numBlocks((SIZE+BLOCKSIZE-1)/BLOCKSIZE);

            T hostPartialResult[numBlocks.x];
            T* devPartialResult;
            size_t partialResultSize = numBlocks.x*sizeof(T);

            error = cudaMalloc((void**)&devPartialResult, partialResultSize);
            if(error != cudaSuccess) {
                std::cerr << "Failed to allocate partialResult for Vector dot product" << std::endl;
                throw std::runtime_error("Failed to allocate device memory.");
            }

            cudaDotProduct<<<numBlocks, threadsPerBlock>>>(devA, devB, devPartialResult, SIZE);

            error = cudaMemcpy(hostPartialResult, devPartialResult, partialResultSize, cudaMemcpyDeviceToHost);
            if(error != cudaSuccess) {
                std::cerr << "Failed to copy partial result from device to host" << std::endl;
                throw std::runtime_error("Failed to copy host to host memory.");
            }

            for(int i = 0; i < numBlocks.x; i++) {
                result += hostPartialResult[i];
            }

            cudaFree(devA);
            cudaFree(devB);
            cudaFree(devPartialResult);

            return result;
        }

        template<typename T>
        __global__ void cudaDotProduct(const T* devA, const T* devB, T* devPartialResult, int size) {
            __shared__ T s_cache[BLOCKSIZE];
            int index = (blockIdx.x*blockDim.x) + threadIdx.x;

            if(index < size) {
                s_cache[threadIdx.x] = devA[index] * devB[index];
            } else {
                s_cache[threadIdx.x] = 0;
            }

            __syncthreads();

            for(int stride = blockDim.x/2; stride > 0; stride/=2) {
                if(threadIdx.x < stride) {
                    s_cache[threadIdx.x] += s_cache[threadIdx.x + stride];
                }
                __syncthreads();
            }

            if(threadIdx.x == 0) { devPartialResult[blockIdx.x] = s_cache[0]; }
        }



    }
}
