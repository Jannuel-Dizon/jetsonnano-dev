add_library(Vector src/Vector.cu)

target_include_directories(Vector
                          PUBLIC
                          ${CMAKE_CURRENT_SOURCE_DIR})
