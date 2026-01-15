cmake_minimum_required(VERSION 3.16)
project(faceDetect LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

add_executable(faceDetect showFaces.cpp faceDetect.cpp)

target_include_directories(faceDetect PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(faceDetect PRIVATE ${OpenCV_LIBS})