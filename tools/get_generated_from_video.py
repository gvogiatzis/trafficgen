import sys
import cv2
import os

os.mkdir('video')
vidcap = cv2.VideoCapture(sys.argv[1])
success, image = vidcap.read()
count = 0
while success:
    h, w, c = image.shape
    image = image[:, w//2:, :]
    image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f'video/frame{count:06}.png', image)     # save frame as JPEG file
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
