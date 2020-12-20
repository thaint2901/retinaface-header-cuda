import numpy as np
from math import ceil
from itertools import product
import torch
torch.manual_seed(0)
from retinaface._C import decode as decode_cuda

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    sm = e_x / e_x.sum(axis=1)[..., np.newaxis]
    return sm

def decode_np(loc, priors, variances=[0.1, 0.2]):
    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_landm_np(pre, priors, variances=[0.1, 0.2]):
    landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), axis=1)
    return landms

class PriorBox:
    def __init__(self):
        self.min_sizes = [[10, 20], [32, 64], [128, 256]]
        self.steps = [8, 16, 32]
        self.image_size = (540, 960)
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            anchor = []
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchor += [cx, cy, s_kx, s_ky]

            anchor = np.array(anchor).reshape(-1, 4)
            anchors.append(anchor)
        return anchors

top_n = 100
threshold = 0.2
width = 960
height = 540
resize = 0.5
h, w = 17, 30
step = 32
anchors = PriorBox().forward()
scale = np.array([width, height] * 2)
scale1 = np.array([width, height] * 5)

loc = torch.randn((1, 8, h, w), dtype=torch.float).to("cuda")
conf = torch.randn((1, 4, h, w), dtype=torch.float).to("cuda")
landms = torch.randn((1, 20, h, w), dtype=torch.float).to("cuda")
priors = torch.tensor([[10, 20], [32, 64], [128, 256]], dtype=torch.float)

scores_cuda, boxes_cuda, landms_cuda = decode_cuda(loc.float(), conf.float(), landms.float(), priors[2].view(-1).tolist(), width, height, resize, step, threshold, top_n)
scores_cuda = scores_cuda.cpu().numpy()[0]
boxes_cuda = boxes_cuda.cpu().numpy()[0]
landms_cuda = landms_cuda.cpu().numpy()[0]

loc = loc.cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, 4)
conf = conf.cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, 2)
landms = landms.cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, 10)

scores = _softmax(conf)[:, 1]
order = scores.argsort()[::-1][:top_n]
scores = scores[order]
boxes = decode_np(loc, anchors[2])
boxes = boxes[order]
boxes = boxes * scale / resize
landms = decode_landm_np(landms, anchors[2])
landms = landms[order]
landms = landms * scale1 / resize

scores_error = np.sum(np.abs(scores - scores_cuda))
boxes_error = np.sum(np.abs(boxes - boxes_cuda))
landms_error = np.sum(np.abs(landms - landms_cuda))
print(scores_error, boxes_error, landms_error)
# print(boxes.cpu())