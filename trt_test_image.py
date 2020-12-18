import numpy as np
import cv2
import timeit
import onnxruntime as rt
import torch
from retinaface._C import decode as decode_cuda


weight_paths = "/vscode/retinaface/faceDetector_540_960_byte_t.onnx"
ort_session = rt.InferenceSession(weight_paths)
input_name = ort_session.get_inputs()[0].name

image_path = "/vscode/retinaface/test.png"
net_inshape = (540, 960)  # h, w
im_height, im_width = net_inshape
rgb_mean = (104, 117, 123) # bgr order

image = cv2.imread(image_path)
h, w = image.shape[:2]
resize = float(net_inshape[1]) / float(w)
batch_size = 1
img = cv2.resize(image, net_inshape[::-1])
img = np.float32(img)
img -= rgb_mean
img = img.transpose(2, 0, 1)
img = np.stack([img] * batch_size)

top_n = 100
threshold = 0.2
width = 960
height = 540
resize = 0.5

onnx_output = ort_session.run(None, {input_name: img})
for i in range(3):
    priors = torch.tensor([[10, 20], [32, 64], [128, 256]], dtype=torch.float)
    loc = torch.from_numpy(onnx_output[i]).to("cuda")
    conf = torch.from_numpy(onnx_output[3 + i]).to("cuda")
    landms = torch.from_numpy(onnx_output[6 + i]).to("cuda")

    scores, boxes, landms = decode_cuda(loc.float(), conf.float(), landms.float(), priors.view(-1).tolist(), width, height, resize, threshold, top_n)
    print(scores.cpu().flatten())