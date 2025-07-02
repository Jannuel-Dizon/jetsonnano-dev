#include "HelloWorld_CUDA.h"
#include <iostream>

__global__ void CUDA_Test() {
    printf("Hello World from GPU!\n\r");
}

void HelloWorld_CUDA() {
    CUDA_Test<<<1, 1>>>();

    cudaDeviceSynchronize();
}
