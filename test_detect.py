import numpy as np
import cv2
import timeit
import torch
from retinaface._C import Engine
from imutils.video import VideoStream

model = Engine.load("/retinaface/pretrained/faceDetector_540_960_batch_decode_nms.plan")

image_path = "images/test.png"
vs = VideoStream("rtsp://admin:meditech123@192.168.101.99:554").start()
net_inshape = (540, 960)  # h, w
im_height, im_width = net_inshape
rgb_mean = (104, 117, 123) # bgr order
threshold = 0.95
detections_per_im = 50
batch_size = 1
while True:
    # image = cv2.imread(image_path)
    image = vs.read()
    image = image.copy()
    h, w = image.shape[:2]
    resize = float(net_inshape[1]) / float(w)
    img = cv2.resize(image, net_inshape[::-1])
    img = np.float32(img)
    img -= rgb_mean
    img = img.transpose(2, 0, 1)
    img = np.stack([img] * batch_size)

    t0 = timeit.default_timer()
    input_tensor = torch.from_numpy(img.copy()).to("cuda")
    scores_cuda, boxes_cuda, landms_cuda = model(input_tensor)
    scores = scores_cuda.cpu().numpy()[0]
    boxes = boxes_cuda.cpu().numpy()[0]
    landms = landms_cuda.cpu().numpy()[0]
    t1 = timeit.default_timer()
    print(t1-t0)

    for i in range(50):
        score = scores[i]
        if score < threshold:
            break
        startX, startY, endX, endY = boxes[i]
        cv2.rectangle(image, (startX, startY), (endX, endY), (0,0,255), 2)
    
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xff
    if key == ord("q"):
        break

cv2.destroyAllWindows()