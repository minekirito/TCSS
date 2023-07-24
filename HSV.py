import cv2
import numpy as np
import math
from PIL import Image
import pandas as pd

#6维数据 averageH, averageS, averageV, stdH, stdS, stdV

for i in range(len(data)):
    imgpath = 'D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/web/web'  + str(i + 1) + '.jpg'
    # imgpath = 'D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/avatar/avatar'  + str(i + 1) + '.jpg'
    img = Image.open(imgpath)
    if img.mode == 'P':  # 必须是RGB模式 P是GIF的格式
        img = img.convert('RGB')
        img.save(
            'D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/web/web' + str(i + 1) + '.jpg')
            # 'D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/avatar/avatar' + str(i + 1) + '.jpg')

    image = cv2.imread(imgpath)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(img_hsv)
    averageH = np.average(H)
    averageS = np.average(S)
    averageV = np.average(V)
    stdH = np.std(H)
    stdS = np.std(S)
    stdV = np.std(V)
    list = averageH, averageS, averageV, stdH, stdS, stdV

    data = pd.DataFrame(list)
    data = data.T
    data.to_csv('HSV_web_steam_augmented.csv', mode='a', header=False)
    # data.to_csv('HSV_steam_augmented.csv', mode='a', header=False)

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

        image = cv2.imread(imgpath)
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(img_hsv)
        averageH = np.average(H)
        averageS = np.average(S)
        averageV = np.average(V)
        stdH = np.std(H)
        stdS = np.std(S)
        stdV = np.std(V)
        list = averageH, averageS, averageV, stdH, stdS, stdV
        print(str(m) + str(i) + '计算完成', list)
        # data = np.array(list)

        data = pd.DataFrame(list)
        data = data.T
        data.to_csv('HSV_web_steam_augmented.csv', mode='a', header=False)
        # data.to_csv('HSV_steam_augmented.csv', mode='a', header=False)
