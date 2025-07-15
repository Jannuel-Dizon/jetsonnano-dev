#ifndef VECTOR_H
#define VECTOR_H

#include "Matrix.h"
#include <concepts>
#include <stdexcept>

namespace MyLA {
    namespace Static {
        template <typename T, int A_ROWS, int A_COLS>
            requires std::is_arithmetic<T>::value
        class Matrix;

        template<typename T>
        __global__ void cudaVectorSum(const T* devA, const T* devB, T* devC, int size);

        template<typename T>
        __global__ void cudaDotProduct(const T* devA, const T* devB, T* devPartialResult, int size);

        template<typename T>
        __global__ void cudaOuterProduct(const T* devA, const T* devB, T* devC, int sizeA, int sizeB);

        template<typename T>
        __global__ void cudaL2Norm(const T* devA, T* devPartialResult, int size);

        template<typename T>
        __global__ void cudaTranspose(const T* devA, T* devB, int size);

        template<typename T>
        __global__ void cudaScaleVector(const T scalar, const T* devA, T* devB, int size);

        template <typename T, int SIZE>
            requires std::is_arithmetic<T>::value
        class Vector {
        private:
            T hostData[SIZE];
        public:
            Vector<T, SIZE>(T defaultVal = 0);
            int getSize() const { return SIZE; }
            T* getData() { return hostData; }
            const T* getData() const { return hostData; }
            T& operator[](int index) {
                if(index < 0 || index >= SIZE) {
                    throw std::runtime_error("Access to Vector is out of bounds");
                }
                return hostData[index];
            }
            const T& operator[](int index) const {
                if(index < 0 || index >= SIZE) {
                    throw std::runtime_error("Access to Vector is out of bounds");
                }
                return hostData[index];
            }

            // Vector operations
            T l2Norm() const;
            Matrix<T, 1, SIZE> transpose() const;
            template<int B_SIZE>
            Vector<T, SIZE> operator+(const Vector<T, B_SIZE>& hostB) const;
            T dotProduct(const Vector<T, B_SIZE>& hostB) const;
            Matrix<T, SIZE, B_SIZE> outerProduct(const Vector<T, B_SIZE>& hostB) const;
        };

        template<typename T, int SIZE>
        Vector<T, SIZE> operator*(T scalar, const Vector<T, SIZE>& hostA);


    } // namespace Static
} // namespace MyLA

#endif // VECTOR_H
