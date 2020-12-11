import numpy as np

# Generates anchors for export.cpp

anchors = np.load("./prior_540_960.npy").astype(np.float32)

axis = str(np.round(anchors.astype(np.float64).flatten(), decimals=2).tolist()).replace('[', '{').replace(']', '}').replace('}, ', '},\n')

print("Axis-aligned:\n"+axis+'\n')
