bbox + landmarks decode and nms using cuda

# Dependence

* Protoc

```bash
docker run --gpus all -itd --ipc=host --privileged --name face_recognition -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /mnt/sda2/ExternalHardrive/face_recognition:/face_recognition \                
    nvcr.io/nvidia/pytorch:20.03-py3
```

CPP: Chứa phần export và infer engine từ onnx bằng cpp

csrc: Chứa thư viện ccuda và cpp

Cài thư viện:
```
python setup.py install
```

# Jetson

docker run --runtime nvidia -itd --net=host --privileged --name deepstream -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /home/mdt/thaint/retinaface-header-cuda:/retinaface-header-cuda \
    nvcr.io/nvidia/deepstream-l4t:5.0.1-20.09-samples

apt-get install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \
   libgstrtspserver-1.0-dev libx11-dev

apt-get install build-essential