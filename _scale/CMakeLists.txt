cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(_scale)

find_package(Torch REQUIRED)

include_directories(/usr/include/python3.8)

# Define our library target
add_library(_scale SHARED op.cpp)
# Enable C++14
target_compile_features(_scale PRIVATE cxx_std_14)
# Link against LibTorch
target_link_libraries(_scale "${TORCH_LIBRARIES}" python3.8)