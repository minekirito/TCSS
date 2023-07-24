import math
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd

def grey_entropy(image):
    # image = cv2.imread(r"D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/avatar/avatar2.jpg", 0)
    rows, cols = image.shape[:2]
    gray_hist = np.zeros([256], np.uint64)
    for i in range(rows):
        for j in range(cols):
            gray_hist[image[i][j]] += 1
    # 归一化灰度直方图，即概率直方图
    normGrayHist = gray_hist / float(rows * cols)
    # 累加直方图
    zeroCumuMoment = np.zeros(256, np.float32)
    for i in range(256):
        if i == 0:
            zeroCumuMoment[i] = normGrayHist[i]
        else:
            zeroCumuMoment[i] = zeroCumuMoment[i - 1] + normGrayHist[i]

    # 计算灰度级的熵
    entropy = np.zeros(256, np.float32)
    for i in range(256):
        if i == 0:
            if normGrayHist[i] == 0:
                entropy[i] = 0
            else:
                entropy[i] = - normGrayHist[i] * math.log10(normGrayHist[i])
        else:
            if normGrayHist[i] == 0:
                entropy[i] = entropy[i - 1]
            else:
                entropy[i] = entropy[i - 1] - normGrayHist[i] * math.log10(normGrayHist[i])
    # 阈值计算
    fT = np.zeros(256, np.float32)
    ft1, ft2 = 0.0, 0.0
    totalEntroy = entropy[255]
    print(totalEntroy)
    return totalEntroy

if __name__ == "__main__":
    for i in range(len(data)):
        imgpath = 'D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/web/web' + str(i + 1) + '.jpg'
        # imgpath = 'D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/avatar/avatar' + str(i + 1) + '.jpg'
        img = Image.open(imgpath)
        if img.mode == 'P':  # 必须是RGB模式 P是GIF的格式
            img = img.convert('RGB')
            img.save('D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/web/web' + str(i + 1) + '.jpg')
            # img.save('D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/avatar/avatar' + str(i + 1) + '.jpg')
        src = cv2.imread(imgpath)
        totalEntroy = grey_entropy(src)
        data = pd.DataFrame([totalEntroy])
        data = data.T
        data.to_csv('Entropy_web_steam_augmented.csv', mode='a', header=False)
        # data.to_csv('Entropy_steam_augmented.csv', mode='a', header=False)

    text = ['blur', 'brightness']

    for m in text:
        for i in range(len(data)):
            imgpath = 'D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/web/web' + m + str(i + 1) + '.jpg'
            # imgpath = 'D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/avatar/avatar' + m + str(i + 1) + '.jpg'
            img = Image.open(imgpath)
            if img.mode == 'P':  # 必须是RGB模式 P是GIF的格式
                img = img.convert('RGB')
                img.save('D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/web/web' + m + str(i + 1) + '.jpg')
                # img.save('D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/avatar/avatar' + m + str(i + 1) + '.jpg')

            src = cv2.imread(imgpath)
            totalEntroy = grey_entropy(src)
            data = pd.DataFrame([totalEntroy])
            data = data.T
            data.to_csv('Entropy_web_steam_augmented.csv', mode='a', header=False)
            # data.to_csv('Entropy_steam_augmented.csv', mode='a', header=False)
