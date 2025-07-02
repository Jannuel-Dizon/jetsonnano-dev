#include <iostream>
#include "HelloWorld_CUDA.h"

int main(void) {

    std::cout << "Hello World!" << std::endl;

    HelloWorld_CUDA();

    std::cout << "Kernel launched" << std::endl;

    return 0;
}
