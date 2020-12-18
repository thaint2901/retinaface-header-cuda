# import torch
# import timeit
# from retinaface._C import decode as decode_cuda
# from retinaface._C import nms as nms_cuda

# import numpy as np

# priors = np.load("./prior_540_960.npy").astype(np.float32)
# priors = torch.from_numpy(priors)

# loc = np.load("loc.npy").astype(np.float32)
# conf = np.load("conf.npy").astype(np.float32)
# landms = np.load("landms.npy").astype(np.float32)
# loc = torch.from_numpy(loc).to("cuda")
# conf = torch.from_numpy(conf).to("cuda")
# landms = torch.from_numpy(landms).to("cuda")

# top_n = 100
# ndetections = 50
# threshold = 0.2
# width = 960
# height = 540
# resize = 0.5
# nms = 0.4
# # while True:
# # for _ in range(5):
# t0 = timeit.default_timer()
# scores, boxes, landms = decode_cuda(loc.float(), conf.float(), landms.float(), priors.view(-1).tolist(), width, height, resize, threshold, top_n)
# t1 = timeit.default_timer()

# nms_scores, nms_boxes, nms_landms = nms_cuda(scores.float(), boxes.float(), landms.float(), nms, ndetections)
# print(t1-t0)
# print(scores.cpu())

import numpy as np
import torch
torch.manual_seed(0)
from retinaface._C import decode as decode_cuda

top_n = 100
threshold = 0.2
width = 960
height = 540
resize = 0.5
h, w = 17, 30

loc = torch.randn((1, 8, h, w), dtype=torch.float).to("cuda")
conf = torch.randn((1, 4, h, w), dtype=torch.float).to("cuda")
landms = torch.randn((1, 20, h, w), dtype=torch.float).to("cuda")
priors = torch.tensor([[10, 20], [32, 64], [128, 256]], dtype=torch.float)


scores, boxes, landms = decode_cuda(loc.float(), conf.float(), landms.float(), priors.view(-1).tolist(), width, height, resize, threshold, top_n)

print(scores.cpu())
# print(boxes.cpu())