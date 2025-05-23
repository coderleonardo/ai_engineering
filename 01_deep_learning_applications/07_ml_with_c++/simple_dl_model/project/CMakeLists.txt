cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(nn_libtorch CXX)

# Defina o padrão C++ explicitamente
set(CMAKE_CXX_STANDARD 17) # LibTorch geralmente prefere C++17 ou C++14
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Garante que não usaremos extensões GNU que possam causar incompatibilidade

# Onde você extraiu a LibTorch. Ajuste este caminho!
set(CMAKE_PREFIX_PATH "./libtorch")

find_package(Torch REQUIRED)

# Adiciona flags do compilador C++ exigidas pela LibTorch.
# Garante a compatibilidade da ABI C++11.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")

add_executable(nn_app main.cpp)
target_link_libraries(nn_app "${TORCH_LIBRARIES}")