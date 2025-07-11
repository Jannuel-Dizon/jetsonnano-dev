#ifndef MATRIX_H
#define MATRIX_H

namespace MyLA {
    template <typename T>
    __global__ void matAddKernel(T* devA, T* devB, T* devC, int rows, int cols);

    template <typename T>
    __global__ void matmatMulKernel(T* devA, T* devB, T* devC, int rowsA, int colsAB, int rowsB);

    template <typename T>
    __global__ void matvecMulKernel(T* devA, T* devB, T* devC, int rows, int cols);

    template <typename T>
    __global__ void transposeKernel(T* devA);

    template <typename T, int ROWS, int COLS>
        requires std::is_arithmetic<T>::value
    class Matrix {
    private:
        T hostData[ROWS*COLS];
        class tempRow {
        private:
            Matrix* ptrMatrix;
            int tRow;
        public:
            tempRow(Matrix* tmpMatrix, int tmpRow) : prtMatrix(tmpMatrix), tRow(tmpRow) {}

            T& operator[](int col) {
                assert(col >= 0 && col < COLS);
                return ptrMatrix->hostData[tRow*COLS + col];
            }
        };
    public:
        Matrix(T defaultVal = 0);
        Matrix<T, ROWS, COLS> operator+(const Matrix<T, ROWS, COLS>& hostB);
        template<int B_ROWS, int B_COLS>
        Matrix<T, ROWS, B_COLS> operator*(const Matrix<T, B_ROWS, B_COLS>& hostB);
        Vector<T, SIZE> operator*(const Vector<T, SIZE>& hostB);
        void transpose();
        tempRow operator[](int row);
        int getRows() const { return ROWS; }
        int getCols() const { return COLS; }
        const T* getData() const { return hostData; }
        T* getData() { return hostData; }
    };
}

#endif
