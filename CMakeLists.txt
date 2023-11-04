cmake_minimum_required(VERSION 3.26)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-12)
set(CMAKE_CUDA_COMPILER /usr/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 75)

enable_language(CUDA)
project(XorAI LANGUAGES CXX CUDA)

find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)
link_libraries(${JSONCPP_LIBRARIES})

set(PROJECT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${PROJECT_DIR}/lib/)
include_directories(${PROJECT_DIR}/src/)

file(GLOB_RECURSE HEADERS ${PROJECT_DIR}/lib/*.h)
file(GLOB_RECURSE SOURCES ${PROJECT_DIR}/src/*.cpp)
file(GLOB_RECURSE CUDA_HEADERS ${PROJECT_DIR}/lib/*.cuh)
file(GLOB_RECURSE CUDA_SOURCES ${PROJECT_DIR}/src/*.cu)

add_executable(XorAI ${PROJECT_DIR}/main.cpp ${HEADERS} ${CUDA_HEADERS} ${SOURCES} ${CUDA_SOURCES})
#add_executable(XorAI ${PROJECT_DIR}/main.cpp ${HEADERS} ${SOURCES})

# Remove -lquadmath if you do not plan on using the __float128 type and don't have the quadmath C binaries.
target_link_libraries(XorAI ${JSONCPP_LIBRARIES} -lquadmath)
set_target_properties(XorAI PROPERTIES CUDA_SEPARABLE_COMPILATION ON)