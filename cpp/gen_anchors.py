from itertools import product as product
import numpy as np
from math import ceil


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

# Generates anchors for export.cpp

# anchors = np.load("./prior_540_960.npy").astype(np.float32)
anchors = PriorBox().forward()

axis = str([np.round(anchor.reshape(-1), decimals=2).tolist() for anchor in anchors]).replace('[', '{').replace(']', '}').replace('}, ', '},\n')

# axis = str(np.round(anchors.astype(np.float64).flatten(), decimals=2).tolist()).replace('[', '{').replace(']', '}').replace('}, ', '},\n')

print("Axis-aligned:\n"+axis+'\n')
