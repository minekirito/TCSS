#!/usr/bin/env python
# coding: utf-8

import cv2

# 读取图像
def getImage(img_path):
    image = cv2.imread(img_path)
    return image

# 获取HSV空间
def get_hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # hsv=cv2.resize(hsv,(184,184))
    hsv = cv2.resize(hsv, (int(image.shape[0]/4), int(image.shape[1]/4)))
    print(image.shape[0],image.shape[1])
    return hsv

# 颜色范围设定
# 对应关系：
# color_set=【balck，grey,white,red1，red2(pink),orange,yellow,green,cyan,blue,purple】
#          【0，     1，  2，   3，   4，    5，    6，   7，  8，  9，   10】

def get_color(H, S, V):
    if (H >= 26 and H <= 34 and S >= 43 and S <= 255 and V >= 46 and V <= 255):
        return 6

    if (H >= 35 and H <= 77 and S >= 43 and S <= 255 and V >= 46 and V <= 255):
        return 7

    if (H >= 0 and H <= 180 and S >= 0 and S <= 255 and V >= 0 and V <= 46):
        return 0

    if (H >= 0 and H <= 180 and S >= 0 and S <= 30 and V >= 221 and V <= 255):
        return 2

    if (H >= 0 and H <= 180 and S >= 0 and S <= 43 and V >= 46 and V <= 220):
        return 1

    if (H >= 11 and H <= 25 and S >= 43 and S <= 255 and V >= 46 and V <= 255):
        return 5

    if (H >= 78 and H <= 99 and S >= 43 and S <= 255 and V >= 46 and V <= 255):
        return 8

    if (H >= 100 and H <= 124 and S >= 43 and S <= 255 and V >= 46 and V <= 255):
        return 9

    if (H >= 125 and H <= 155 and S >= 43 and S <= 255 and V >= 46 and V <= 255):
        return 10

    if (H >= 0 and H <= 10 and S >= 43 and S <= 255 and V >= 46 and V <= 255):
        return 3

    if (H >= 156 and H <= 180 and S >= 43 and S <= 255 and V >= 46 and V <= 255):
        return 4
    return 3


def getColor_sum(test_path):
    image = getImage(test_path)
    hsv = get_hsv(image)
    color_num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    h, s, v = cv2.split(hsv)
    for i in range(int(image.shape[1]/4)):
        for j in range(int(image.shape[0]/4)):
            color = get_color(h[i][j], s[i][j], v[i][j])
            color_num[color] = color_num[color] + 1
    print(color_num)
    return color_num


from PIL import Image
import pandas as pd

if __name__ == '__main__':
    for i in range(len(data)):
        # imgpath = 'D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/web/web' + str(i + 1) + '.jpg'
        imgpath = 'D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/avatar/avatar'  + str(i + 1) + '.jpg'
        img = Image.open(imgpath)
        if img.mode == 'P':  # 必须是RGB模式 P是GIF的格式
            img = img.convert('RGB')
            # img.save('D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/web/web' + str(i + 1) + '.jpg')
            img.save('D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/avatar/avatar' + str(i + 1) + '.jpg')

        color_num = getColor_sum(imgpath)
        data = pd.DataFrame(color_num)
        data = data.T
        data.to_csv('Colordname_steam_augmented.csv', mode='a', header=False)

#Data augmentation methods
    text = ['blur', 'brightness']

    for m in text:
        for i in range(len(data)):
            # imgpath = 'D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/web/web' + m + str(i + 1) + '.jpg'
            imgpath = 'D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/avatar/avatar' + m + str(i + 1) + '.jpg'
            img = Image.open(imgpath)
            if img.mode == 'P':  # 必须是RGB模式 P是GIF的格式
                img = img.convert('RGB')
                # img.save('D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/web/web' + m + str(i + 1) + '.jpg')
                img.save('D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/avatar/avatar' + m + str(i + 1) + '.jpg')
            print(imgpath)
            color_num = getColor_sum(imgpath)
            data = pd.DataFrame(color_num)
            data = data.T
            data.to_csv('Colordname_steam_augmented.csv', mode='a', header=False)
