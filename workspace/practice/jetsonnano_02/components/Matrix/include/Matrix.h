#ifndef MATRIX_H
#define MATRIX_H

namespace MyLA {
    template <typename T>
    __global__ void matAddKernel(T* devA, T* devB, T* devC, int rows, int cols);

    template <typename T>
    __global__ void matmatMulKernel(T* devA, T* devB, T* devC, int rowsA, int colsrowsAB, int colsB);

    template <typename T>
    __global__ void matvecMulKernel(T* devA, T* devB, T* devC, int rows, int cols);

    template <typename T>
    __global__ void transposeKernel(T* devA);

    template <typename T, int A_ROWS, int A_COLS>
        requires std::is_arithmetic<T>::value
    class Matrix {
    private:
        T hostData[A_ROWS*A_COLS];
        class tempRow {
        private:
            Matrix* ptrMatrix;
            int tRow;
        public:
            tempRow(Matrix* tmpMatrix, int tmpRow) : ptrMatrix(tmpMatrix), tRow(tmpRow) {}

            T& operator[](int col) {
                assert(col >= 0 && col < A_COLS);
                return ptrMatrix->hostData[tRow*A_COLS + col];
            }
        };
    public:
        Matrix(T defaultVal = 0);
        Matrix<T, A_ROWS, A_COLS> operator+(const Matrix<T, A_ROWS, A_COLS>& hostB);
        template<int B_ROWS, int B_COLS>
        Matrix<T, A_ROWS, B_COLS> operator*(const Matrix<T, B_ROWS, B_COLS>& hostB);
        template<int SIZE>
        Vector<T, A_ROWS> operator*(const Vector<T, SIZE>& hostB);
        void transpose();
        tempRow operator[](int row) {
            assert(row >= 0 && row < A_ROWS);
            return tempRow(this, row);
        }
        int getRows() const { return A_ROWS; }
        int getCols() const { return A_COLS; }
        const T* getData() const { return hostData; }
        T* getData() { return hostData; }
    };
}

#endif
