cmake_minimum_required(VERSION 3.16)
project(Video-Special-Effects LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

add_executable(project1 vidDisplay.cpp filter.cpp faceDetect/faceDetect.cpp)

target_include_directories(project1 PUBLIC "${CMAKE_SOURCE_DIR}/faceDetect" PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(project1 PRIVATE ${OpenCV_LIBS})

# ONNX Runtime (Homebrew)
find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
        PATHS /opt/homebrew/include /usr/local/include
        PATH_SUFFIXES onnxruntime
)

find_library(ORT_LIB onnxruntime
        PATHS /opt/homebrew/lib /usr/local/lib
)

if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIB)
    message(FATAL_ERROR "Could not find ONNX Runtime. Check: brew install onnxruntime")
endif()

target_include_directories(project1 PRIVATE ${ORT_INCLUDE_DIR})
target_link_libraries(project1 PRIVATE ${ORT_LIB})