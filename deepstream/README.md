mkdir build && cd build
cmake -DDeepStream_DIR=/opt/nvidia/deepstream/deepstream-5.0 \
    -DCMAKE_CUDA_FLAGS="--expt-extended-lambda -std=c++14" ..