# import numpy as np
# import cv2
# import timeit
# import onnxruntime as rt
# import torch
# from retinaface._C import decode as decode_cuda
# from retinaface._C import Engine


# weight_paths = "/nvidia/retinaface-header-cuda/pretrained/faceDetector_540_960_batch.onnx"
# ort_session = rt.InferenceSession(weight_paths)
# input_name = ort_session.get_inputs()[0].name

# image_path = "/nvidia/retinaface-header-cuda/images/test.png"
# net_inshape = (540, 960)  # h, w
# im_height, im_width = net_inshape
# rgb_mean = (104, 117, 123) # bgr order

# image = cv2.imread(image_path)
# h, w = image.shape[:2]
# resize = float(net_inshape[1]) / float(w)
# batch_size = 1
# img = cv2.resize(image, net_inshape[::-1])
# img = np.float32(img)
# img -= rgb_mean
# img = img.transpose(2, 0, 1)
# img = np.stack([img] * batch_size)

# top_n = 250
# score_thresh = 0.2
# width = 960
# height = 540
# resize = 0.5
# priors = torch.tensor([[10, 20], [32, 64], [128, 256]], dtype=torch.float)
# steps = [8, 16, 32]

# model = Engine.load("/nvidia/retinaface-header-cuda/pretrained/faceDetector_540_960_batch_decode.plan")

# input_tensor = torch.from_numpy(img.copy()).to("cuda")
# scores_cuda, boxes_cuda, landms_cuda = model(input_tensor)
# scores_cuda = scores_cuda.cpu().numpy()[0]
# boxes_cuda = boxes_cuda.cpu().numpy()[0]
# landms_cuda = landms_cuda.cpu().numpy()[0]

# onnx_output = ort_session.run(None, {input_name: img})
# scores_, boxes_, landms_ = [], [], []
# for i in range(3):
#     loc = torch.from_numpy(onnx_output[i]).to("cuda")
#     conf = torch.from_numpy(onnx_output[3 + i]).to("cuda")
#     landms = torch.from_numpy(onnx_output[6 + i]).to("cuda")

#     decode = decode_cuda(loc.float(), conf.float(), landms.float(), priors[i].view(-1).tolist(), width, height, resize, steps[i], score_thresh, top_n)
#     scores_.append(decode[0].cpu().numpy()[0])
#     boxes_.append(decode[1].cpu().numpy()[0])
#     landms_.append(decode[2].cpu().numpy()[0])

# scores_ = np.concatenate(scores_, axis=0)
# boxes_ = np.concatenate(boxes_, axis=0)
# landms_ = np.concatenate(landms_, axis=0)

# print(scores_-scores_cuda)

