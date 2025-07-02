add_library(HelloWorld_CUDA src/HelloWorld_CUDA.cu)

target_include_directories(HelloWorld_CUDA
                           PUBLIC
                           ${CMAKE_CURRENT_SOURCE_DIR}/include)
