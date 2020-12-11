cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(retinaface LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
find_package(CUDA REQUIRED)
enable_language(CUDA)
find_package(OpenCV REQUIRED)

if(DEFINED TensorRT_DIR)
  include_directories("${TensorRT_DIR}/include")
  link_directories("${TensorRT_DIR}/lib")
endif(DEFINED TensorRT_DIR)
include_directories(${CUDA_INCLUDE_DIRS})

add_library(retinaface SHARED
  ../csrc/cuda/decode.h
  ../csrc/cuda/decode.cu
  ../csrc/cuda/nms.h
  ../csrc/cuda/nms.cu
  ../csrc/cuda/utils.h
  ../csrc/engine.h
  ../csrc/engine.cpp
)
set_target_properties(retinaface PROPERTIES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(retinaface PUBLIC nvinfer nvonnxparser ${OpenCV_LIBS} cuda ${CUDA_LIBRARIES})

add_executable(export export.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(export PRIVATE retinaface ${OpenCV_LIBS})

add_executable(infer infer.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(infer PRIVATE retinaface ${OpenCV_LIBS} cuda ${CUDA_LIBRARIES})
