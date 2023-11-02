cmake_minimum_required(VERSION 3.26)

project(XorAI)

set(CMAKE_CXX_STANDARD 20)

find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)
link_libraries(${JSONCPP_LIBRARIES})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}src/)

file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/lib/*.h)
file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_executable(XorAI ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp ${HEADERS} ${SOURCES})
target_link_libraries(XorAI ${JSONCPP_LIBRARIES})