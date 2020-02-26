import numpy as np
import os
import cv2


source_path = "/home/pengzhang/dataset/CASIA-CeFA/phase1/train/"
save_path = "/home/pengzhang/dataset/CASIA-CeFA/phase1/train_crop/"
file_list = os.listdir(source_path)
print(file_list)
for img_dir in file_list:
    file_names = os.listdir(source_path + img_dir + '/profile/')
    save_img_path = save_path + img_dir + '/profile/'
    print(file_names)
    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)
    for i in range(len(file_names)):
        print(file_names[i])
        img = cv2.imread(source_path + img_dir + '/profile/' + str(file_names[i]))
        img = cv2.medianBlur(img, 5)
        # img = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_RGB2BGR)
        b = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
        binary_image = b[1]
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
        print(binary_image.shape)

        x = binary_image.shape[0]  # 高度
        y = binary_image.shape[1]  # 宽度
        edges_x = []
        edges_y = []
        for k in range(x):
            for j in range(y):
                if binary_image[k][j] == 255:
                    edges_x.append(k)
                    edges_y.append(j)
        left = min(edges_x)  # 左边界
        right = max(edges_x)  # 右边界
        width = right - left  # 宽度
        bottom = min(edges_y)  # 底部
        top = max(edges_y)  # 顶部
        height = top - bottom  # 高度

        img = img[left:left + width, bottom:bottom + height]
        # 裁剪区域
        # 眼睛区域：0.25 - 0.5 嘴巴区域：0.65 - 1.0
        x1 = int(0.25 * width)
        x2 = int(0.5 * width)
        x3 = int(0.65 * width)
        img[0:x1, 0:height] = (0, 0, 0)
        img[x2:x3, 0:height] = (0, 0, 0)
        save_name = save_img_path + str(file_names[i])
        img = cv2.resize(img, (224, 224))
        print(img.shape)
        print(save_name)
        cv2.imwrite(save_name, img)
