# cmake_minimum_required(VERSION 3.5.1)

# project(deepstream-retinaface)
# enable_language(CXX)
# include(FindCUDA)

# set(CMAKE_CXX_STANDARD 14)
# find_package(CUDA REQUIRED)
# find_package(OpenCV REQUIRED)

# if(DEFINED TensorRT_DIR)
#   include_directories("${TensorRT_DIR}/include")
#   link_directories("${TensorRT_DIR}/lib")
# endif(DEFINED TensorRT_DIR)
# if(DEFINED DeepStream_DIR)
#   include_directories("${DeepStream_DIR}/sources/includes")
# endif(DEFINED DeepStream_DIR)
# include_directories(${CUDA_INCLUDE_DIRS})

# if(NOT DEFINED ARCH)
#   set(ARCH "sm_75")
# endif(NOT DEFINED ARCH)

# cuda_add_library(nvdsparsebbox_retinaface SHARED
#   ../csrc/cuda/decode.h
#   ../csrc/cuda/decode.cu
#   ../csrc/cuda/nms.h
#   ../csrc/cuda/nms.cu
#   ../csrc/cuda/utils.h
#   ../csrc/engine.cpp
#   nvdsparsebbox_retinaface.cpp
#   OPTIONS -arch ${ARCH} -std=c++14 --expt-extended-lambda
# )
# include_directories(${OpenCV_INCLUDE_DIRS})
# target_link_libraries(nvdsparsebbox_retinaface ${CUDA_LIBRARIES} nvinfer nvinfer_plugin nvonnxparser ${OpenCV_LIBS})


cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(retinaface LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
find_package(CUDA REQUIRED)
enable_language(CUDA)
find_package(OpenCV REQUIRED)
find_package(PkgConfig)
pkg_check_modules(GST REQUIRED gstreamer-1.0>=1.4
                               gstreamer-sdp-1.0>=1.4
                               gstreamer-video-1.0>=1.4
                               gstreamer-app-1.0>=1.4)

if(DEFINED TensorRT_DIR)
  include_directories("${TensorRT_DIR}/include")
  link_directories("${TensorRT_DIR}/lib")
endif(DEFINED TensorRT_DIR)
if(DEFINED DeepStream_DIR)
  include_directories("${DeepStream_DIR}/sources/includes")
  link_directories("${DeepStream_DIR}/lib")
endif(DEFINED DeepStream_DIR)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${GST_INCLUDE_DIRS})

message(STATUS "====================${PLATFORM_TEGRA}")

add_library(nvdsparsebbox_retinaface SHARED
  ../csrc/cuda/decode.h
  ../csrc/cuda/decode.cu
  ../csrc/cuda/nms.h
  ../csrc/cuda/nms.cu
  ../csrc/cuda/utils.h
  ../csrc/engine.cpp
  nvdsparsebbox_retinaface.cpp
)
set_target_properties(nvdsparsebbox_retinaface PROPERTIES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(nvdsparsebbox_retinaface PUBLIC nvinfer nvinfer_plugin ${OpenCV_LIBS} cuda ${CUDA_LIBRARIES})

add_subdirectory(protos)

find_package(Protobuf REQUIRED)
include_directories(${CMAKE_BINARY_DIR}/protos include)

add_library(utils SHARED
  aligner.cpp
  base64.cpp
  utils.cpp
)

add_executable(main main.cpp )
if(DEFINED PLATFORM_TEGRA)
  add_definitions(-DPLATFORM_TEGRA=1)
endif(DEFINED PLATFORM_TEGRA)
target_link_libraries(main PRIVATE utils nvdsparsebbox_retinaface ${OpenCV_LIBS} cuda ${CUDA_LIBRARIES} nvinfer nvinfer_plugin nvonnxparser ${GST_LIBRARIES} nvdsgst_meta nvds_meta nvbufsurface)
target_link_libraries(main PUBLIC protobuf ${Protobuf_LIBRARIES})
target_include_directories(main PUBLIC ${Protobuf_INCLUDE_DIRS})
