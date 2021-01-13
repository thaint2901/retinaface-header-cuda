# Build app + lib

```bash
mkdir build && cd build
cmake -DDeepStream_DIR=/opt/nvidia/deepstream/deepstream-5.0 \
    -DCMAKE_CUDA_FLAGS="--expt-extended-lambda -std=c++14" \
    -DPLATFORM_TEGRA=ON ..
make
```

# Run app

./main rtsp://admin:meditech123@10.68.10.96:554

LD_PRELOAD=build/libnvdsparsebbox_retinaface.so deepstream-app -c configs/ds_config_1vid.txt