import numpy as np
np.random.seed(29)

width = 120
height = 68
x,y = 0,0
a = 0
num_classes = 1
num_anchors = 2
data = np.random.randn(1, 8, 68, 120)


print(data[0,:,y,x])
print([
    data.flatten()[((a * 4 + 0) * height + y) * width + x],
    data.flatten()[((a * 4 + 1) * height + y) * width + x],
    data.flatten()[((a * 4 + 2) * height + y) * width + x],
    data.flatten()[((a * 4 + 3) * height + y) * width + x]
])