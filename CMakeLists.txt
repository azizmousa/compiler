cmake_minimum_required(VERSION 3.1)

project(sketch_builder_compiler)

find_package(OpenCV REQUIRED)

set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")

file(GLOB SRC_PROJECT "src/*.cpp")

set(CMAKE_CXX_FLAGS "-Wall -O2 -g")
add_executable(compiler ${SRC_PROJECT})


target_link_libraries(compiler ${OpenCV_LIBS})

target_include_directories(compiler PRIVATE "${PROJECT_SOURCE_DIR}/include")