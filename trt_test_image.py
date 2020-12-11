import numpy as np
import cv2
import timeit
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda


image_path = "./sample.jpg"
net_inshape = (540, 960)  # h, w
im_height, im_width = net_inshape
rgb_mean = (104, 117, 123) # bgr order

weight_paths = "./faceDetector_540_960_dyn_sim.plan"
trt_logger = trt.Logger(trt.Logger.VERBOSE)
bindings = []
host_inputs = []
cuda_inputs = []
host_outputs = []
cuda_outputs = []
with open(weight_paths, 'rb') as f, trt.Runtime(trt_logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
stream = cuda.Stream()

input_volume = trt.volume((3, im_height, im_width))
batch_size = 1
max_batch_size = engine.max_batch_size
print("max_batch_size: ", max_batch_size)
numpy_array = np.zeros((max_batch_size, input_volume))

for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding)) * max_batch_size
    host_mem = cuda.pagelocked_empty(size, np.float32)
    cuda_mem = cuda.mem_alloc(host_mem.nbytes)
    bindings.append(int(cuda_mem))
    if engine.binding_is_input(binding):
        host_inputs.append(host_mem)
        cuda_inputs.append(cuda_mem)
    else:
        host_outputs.append(host_mem)
        cuda_outputs.append(cuda_mem)
context = engine.create_execution_context()

image = cv2.imread(image_path)
h, w = image.shape[:2]
resize = float(net_inshape[1]) / float(w)
img = cv2.resize(frame, net_inshape[::-1])
img = np.float32(img)
img -= rgb_mean
img = img.transpose(2, 0, 1)
img = np.stack([img] * batch_size)

t0 = timeit.default_timer()
for i in range(batch_size):
    numpy_array[i] = img.ravel()
np.copyto(host_inputs[0], numpy_array.ravel())
cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
context.execute_async(
    batch_size=max_batch_size,
    bindings=bindings,
    stream_handle=stream.handle)

for i in range(len(host_outputs)):
    cuda.memcpy_dtoh_async(host_outputs[i], cuda_outputs[i], stream)
stream.synchronize()

print("{:.5f}".format(timeit.default_timer() - t0))

# for batch size = 1
nms_scores = host_outputs[0].reshape((max_batch_size, -1))[:batch_size]
nms_boxes = host_outputs[2].reshape((max_batch_size, -1, 4))[:batch_size]
nms_landms = host_outputs[1].reshape((max_batch_size, -1, 10))[:batch_size]

print(boxes.shape, scores.shape, landms.shape)

del stream
del cuda_outputs
del cuda_inputs
