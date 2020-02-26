import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import random
from PIL import Image

class EyeMouth(object):

    def __call__(self, img):
        #去除黑边
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        b = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
        binary_image = b[1]
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)

        x = binary_image.shape[0]#高度
        y = binary_image.shape[1]#宽度
        edges_x = []
        edges_y = []
        for i in range(x):
            for j in range(y):
                if binary_image[i][j] == 255:
                    edges_x.append(i)
                    edges_y.append(j)
        if edges_x:
            left = min(edges_x)  # 左边界
            right = max(edges_x)  # 右边界
        else:
            left = 0
            right = x
        if edges_y:
            bottom = min(edges_y)  # 底部
            top = max(edges_y)  # 顶部
        else:
            bottom = 0
            top = y
        width = right - left    #宽度
        height = top - bottom   #高度

        img = img[left:left + width, bottom:bottom + height]
        #裁剪区域 中间涂黑
        #眼睛区域：0.25 - 0.5 嘴巴区域：0.65 - 1.0
        x1 = int(0.25 * width)
        x2 = int(0.5 * width)
        x3 = int(0.65 * width)
        img[0:x1, 0:height] = (0, 0, 0)
        img[x2:x3, 0:height] = (0, 0, 0)

        img = cv2.resize(img, (224, 224))
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        return img

