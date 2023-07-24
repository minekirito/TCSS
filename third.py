import cv2
from PIL import Image
import pandas as pd
import numpy as np

for i in range(len(data)):
    imgpath = 'D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/web/web'  + str(i + 1) + '.jpg'
    # imgpath = 'D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/avatar/avatar'  + str(i + 1) + '.jpg'
    img = Image.open(imgpath)
    if img.mode == 'P':  # 必须是RGB模式 P是GIF的格式
        img = img.convert('RGB')
        img.save('D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/web/web' + str(i + 1) + '.jpg')
        # img.save('D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/avatar/avatar' + str(i + 1) + '.jpg')

    image = cv2.imread(imgpath)
    y, x = image.shape[0:2]
    print(image.shape)# Prints Dimensions of the image
    y2 = y / 3
    x2 = x / 3
    cropped_image = image[int(y2):int(y2 * 2), int(x2):int(x2 * 2)]  # Slicing to crop the image

    img_hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(img_hsv)
    averageH = np.average(H)
    averageS = np.average(S)
    averageV = np.average(V)
    list = averageH, averageS, averageV

    data = pd.DataFrame(list)
    data = data.T
    data.to_csv('thirdHSV_web_steam_augmented.csv', mode='a', header=False)

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
        y, x = image.shape[0:2]
        print(image.shape)  # Prints Dimensions of the image
        y2 = y / 3
        x2 = x / 3
        cropped_image = image[int(y2):int(y2 * 2), int(x2):int(x2 * 2)]  # Slicing to crop the image

        img_hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(img_hsv)
        averageH = np.average(H)
        averageS = np.average(S)
        averageV = np.average(V)
        list = averageH, averageS, averageV
        print(str(m) + str(i) + '计算完成', list)
        # data = np.array(list)

        data = pd.DataFrame(list)
        data = data.T
        data.to_csv('thirdHSV_web_steam_augmented.csv', mode='a', header=False)
        # data.to_csv('thirdHSV_steam_augmented.csv', mode='a', header=False)