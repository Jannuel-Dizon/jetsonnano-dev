add_library(Matrix src/Matrix.cu)

target_include_directories(Matrix
                           PUBLIC
                           ${CMAKE_CURRENT_SOURCE_DIR}/include)
