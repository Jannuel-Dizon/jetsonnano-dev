#include "Vector.h"
#include "Matrix.h"
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <vector>

#define BLOCKSIZE 256
#define BLOCKSIZEX 16
#define BLOCKSIZEY 16

namespace MyLA {
    namespace Static {

        template<typename T, int SIZE>
        MyLA::Static::Vector<T, SIZE>::Vector(T defaultVal) {
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

        template<typename T>
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

            std::vector<T> hostPartialResult((SIZE+BLOCKSIZE-1)/BLOCKSIZE);
            T* devPartialResult;
            size_t partialResultSize = ((SIZE+BLOCKSIZE-1)/BLOCKSIZE)*sizeof(T);

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

            error = cudaMalloc((void**)&devPartialResult, partialResultSize);
            if(error != cudaSuccess) {
                std::cerr << "Failed to allocate partialResult for Vector dot product" << std::endl;
                throw std::runtime_error("Failed to allocate device memory.");
            }

            dim3 threadsPerBlock(BLOCKSIZE);
            dim3 numBlocks((SIZE+BLOCKSIZE-1)/BLOCKSIZE);

            cudaDotProduct<<<numBlocks, threadsPerBlock>>>(devA, devB, devPartialResult, SIZE);

            error = cudaMemcpy(hostPartialResult.data(), devPartialResult, partialResultSize, cudaMemcpyDeviceToHost);
            if(error != cudaSuccess) {
                std::cerr << "Failed to copy partial result from device to host" << std::endl;
                throw std::runtime_error("Failed to copy host to host memory.");
            }

            result = std::accumulate(hostPartialResult.begin(), hostPartialResult.end(), 0);

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


        template<typename T, int SIZE>
        template<int B_SIZE>
        Matrix<T, SIZE, B_SIZE> Vector<T, SIZE>::outerProduct(const Vector<T, B_SIZE>& hostB) const {
            Matrix<T, SIZE, B_SIZE> hostC;

            T* devA;
            T* devB;
            T* devC;
            size_t sizeA = SIZE*sizeof(T);
            size_t sizeB = B_SIZE*sizeof(T);
            size_t sizeC = SIZE*B_SIZE*sizeof(T);

            cudaError_t error;

            error = cudaMalloc((void**)&devA, sizeA);
            if(error != cudaSuccess) {
                std::cerr << "Failed to allocate devA for Vector outer product" << std::endl;
                throw std::runtime_error("Failed to allocate device memory.");
            }

            error = cudaMalloc((void**)&devB, sizeB);
            if(error != cudaSuccess) {
                std::cerr << "Failed to allocate devB for Vector outer product" << std::endl;
                throw std::runtime_error("Failed to allocate device memory.");
            }

            error = cudaMalloc((void**)&devC, sizeC);
            if(error != cudaSuccess) {
                std::cerr << "Failed to allocate devC for Vector outer product" << std::endl;
                throw std::runtime_error("Failed to allocate device memory.");
            }

            error = cudaMemcpy(devA, this->getData(), sizeA, cudaMemcpyHostToDevice);
            if(error != cudaSuccess) {
                std::cerr << "Failed to copy Vector A from host to device" << std::endl;
                throw std::runtime_error("Failed to copy host to device memory.");
            }

            error = cudaMemcpy(devB, hostB.getData(), sizeB, cudaMemcpyHostToDevice);
            if(error != cudaSuccess) {
                std::cerr << "Failed to copy Vector B from host to device" << std::endl;
                throw std::runtime_error("Failed to copy host to device memory.");
            }

            dim3 threadsPerBlock(BLOCKSIZEX, BLOCKSIZEY);
            dim3 numBlocks((B_SIZE+BLOCKSIZEX-1)/BLOCKSIZEX, (SIZE+BLOCKSIZEY-1)/BLOCKSIZEY);

            cudaOuterProduct<<<numBlocks, threadsPerBlock>>>(devA, devB, devC, SIZE, B_SIZE);

            error = cudaMemcpy(hostC.getData(), devC, sizeC, cudaMemcpyDeviceToHost);
            if(error != cudaSuccess) {
                std::cerr << "Failed to copy Vector C from device to host" << std::endl;
                throw std::runtime_error("Failed to copy device to host memory.");
            }

            cudaFree(devA);
            cudaFree(devB);
            cudaFree(devC);

            return hostC;
        }

        template<typename T>
        __global__ void cudaOuterProduct(const T* devA, const T* devB, T* devC, int sizeA, int sizeB) {
            int row = (blockIdx.y*blockDim.y) + threadIdx.y;
            int col = (blockIdx.x*blockDim.x) + threadIdx.x;

            if(row < sizeA && col <sizeB) {
                int index = (row*sizeB) + col;
                devC[index] = devA[row] * devB[col];
            }
        }

        template<typename T, int SIZE>
        T Vector<T, SIZE>::l2Norm() const {
            T result = 0;

            T* devA;
            size_t size = SIZE*sizeof(T);

            std::vector<T> hostPartialResult((SIZE+BLOCKSIZE-1)/BLOCKSIZE);
            T* devPartialResult;
            size_t sizePartialResult = (SIZE+BLOCKSIZE-1)/BLOCKSIZE*sizeof(T);

            cudaError_t error;

            error = cudaMalloc((void**)&devA, size);
            if(error != cudaSuccess) {
                std::cerr << "Failed to allocate devA for Vector l2 normalization" << std::endl;
                throw std::runtime_error("Failed to allocate device memory.");
            }

            error = cudaMalloc((void**)&devPartialResult, sizePartialResult);
            if(error != cudaSuccess) {
                std::cerr << "Failed to allocate partial result for Vector l2 normalization" << std::endl;
                throw std::runtime_error("Failed to allocate device memory.");
            }

            error = cudaMemcpy(devA, this->getData(), size, cudaMemcpyHostToDevice);
            if(error != cudaSuccess) {
                std::cerr << "Failed to copy Vector A from host to device" << std::endl;
                throw std::runtime_error("Failed to copy host to device memory.");
            }

            dim3 threadsPerBlock(BLOCKSIZE);
            dim3 numBlocks((SIZE+BLOCKSIZE-1)/BLOCKSIZE);

            cudaL2Norm<<<numBlocks, threadsPerBlock>>>(devA, devPartialResult, SIZE);

            error = cudaMemcpy(hostPartialResult.data(), devPartialResult, sizePartialResult, cudaMemcpyDeviceToHost);
            if(error != cudaSuccess) {
                std::cerr << "Failed to copy partial result from device to host" << std::endl;
                throw std::runtime_error("Failed to copy device to host memory.");
            }

            result = std::accumulate(hostPartialResult.begin(), hostPartialResult.end(), 0);

            cudaFree(devA);
            cudaFree(devPartialResult);

            return sqrt(result);
        }

        template<typename T>
        __global__ void cudaL2Norm(const T* devA, T* devPartialResult, int size) {
            int index = (blockIdx.x*blockDim.x) + threadIdx.x;
            __shared__ T s_cache[BLOCKSIZE];

            if(index < size) {
                s_cache[threadIdx.x] = devA[index]*devA[index];
            } else {
                s_cache[threadIdx.x] = 0;
            }
            __syncthreads();
            for(int stride = BLOCKSIZE/2; stride > 0; stride /= 2) {
                if(threadIdx.x < stride) {
                    s_cache[threadIdx.x] += s_cache[threadIdx.x+stride];
                }
                __syncthreads();
            }

            if(threadIdx.x == 0) { devPartialResult[blockIdx.x] = s_cache[0]; }
        }

        template<typename T, int SIZE>
        Matrix<T, 1, SIZE> Vector<T, SIZE>::transpose() const {
            Matrix<T, 1, SIZE> hostB;

            T* devA;
            T* devB;
            size_t size = SIZE*sizeof(T);

            cudaError_t error;

            error = cudaMalloc((void**)&devA, size);
            if(error != cudaSuccess) {
                std::cerr << "Failed to allocate devA for Vector transpose" << std::endl;
                throw std::runtime_error("Failed to allocate device memory.");
            }

            error = cudaMalloc((void**)&devB, size);
            if(error != cudaSuccess) {
                std::cerr << "Failed to allocate devB for Vector transpose" << std::endl;
                throw std::runtime_error("Failed to allocate device memory.");
            }

            error = cudaMemcpy(devA, this->getData(), size, cudaMemcpyHostToDevice);
            if(error != cudaSuccess) {
                std::cerr << "Failed to copy Vector A from host to device" << std::endl;
                throw std::runtime_error("Failed to copy host to device memory.");
            }

            dim3 threadsPerBlock(BLOCKSIZE);
            dim3 numBlocks((SIZE+BLOCKSIZE-1)/BLOCKSIZE);

            cudaTranspose<<<numBlocks, threadsPerBlock>>>(devA, devB, SIZE);

            error = cudaMemcpy(hostB,getData(), devB, size, cudaMemcpyDeviceToHost);
            if(error != cudaSuccess) {
                std::cerr << "Failed to copy Vector B from device to host" << std::endl;
                throw std::runtime_error("Failed to copy device to host memory.");
            }

            cudaFree(devA);
            cudaFree(devB);

            return hostB;
        }

        template<typename T>
        __global__ void cudaTranspose(const T* devA, T* devB, int size) {
            int index = (blockIdx.x*blockDim.x) + threadIdx.x;

            if(index < size) {
                devB[index] = devA[index];
            }
        }

        template<typename T, int SIZE>
        Vector<T, SIZE> operator*(T scalar, const Vector<T, SIZE>& hostA) {
            Vector<T, SIZE> hostB;

            T* devA;
            T* devB;
            size_t size = SIZE*sizeof(T);

            cudaError_t error;

            error = cudaMalloc((void**)&devA, size);
            if(error != cudaSuccess) {
                std::cerr << "Failed to allocate devA for scaling Vector" << std::endl;
                throw std::runtime_error("Failed to allocate device memory.");
            }

            error = cudaMalloc((void**)&devB, size);
            if(error != cudaSuccess) {
                std::cerr << "Failed to allocate devB for scaling Vector" << std::endl;
                throw std::runtime_error("Failed to allocate device memory.");
            }

            error = cudaMemcpy(devA, hostA.getData(), size, cudaMemcpyHostToDevice);
            if(error != cudaSuccess) {
                std::cerr << "Failed to copy Vector A from host to device" << std::endl;
                throw std::runtime_error("Failed to copy host to device memory.");
            }

            dim3 threadsPerBlock(BLOCKSIZE);
            dim3 numBlocks((SIZE+BLOCKSIZE-1)/BLOCKSIZE);


            cudaScaleVector<<<numBlocks, threadsPerBlock>>>(scalar, devA, devB, SIZE);

            error = cudaMemcpy(hostB.getData(), devB, size, cudaMemcpyDeviceToHost);
            if(error != cudaSuccess) {
                std::cerr << "Failed to copy Vector B from device to host" << std::endl;
                throw std::runtime_error("Failed to copy device to host memory.");
            }

            cudaFree(devA);
            cudaFree(devB);

            return hostB;
        }

        template<typename T>
        __global__ void cudaScaleVector(const T scalar, const T* devA, T* devB, int size) {
            int index = (blockIdx.x*blockDim.x) + threadIdx.x;

            if(index < size) {
                devB[index] = scalar*devA[index];
            }
        }
    }
}
