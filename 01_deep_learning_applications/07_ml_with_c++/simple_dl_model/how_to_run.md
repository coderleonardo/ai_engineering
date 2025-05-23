Create a CMakeLists.txt like:

```
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(TorchNeuralNetwork)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(torch_nn main.cpp)
target_link_libraries(torch_nn "${TORCH_LIBRARIES}")
set_property(TARGET torch_nn PROPERTY CXX_STANDARD 17)
```

Run in bash: 
```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch .. # libtorch path
cmake --build .
```