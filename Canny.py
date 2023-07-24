# 导包
import cv2 as cv
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd


#method1
def area1(image):
    list=[]
    h,w=image.shape[0],image.shape[1]
    for i in range(h):
        for j in range(w):
                if image[i][j]!=0:
                    coor=(j,i)
                    list.append(coor)
    return len(list)

for i in range(len(data)):
    imgpath = 'D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/web/web'  + str(i + 1) + '.jpg'
    # imgpath = 'D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/avatar/avatar'  + str(i + 1) + '.jpg'
    img = Image.open(imgpath)
    if img.mode == 'P':  # 必须是RGB模式 P是GIF的格式
        img = img.convert('RGB')
        img.save('D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/web/web' + str(i + 1) + '.jpg')
        # img.save('D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/avatar/avatar' + str(i + 1) + '.jpg')

    image = cv.imread(imgpath)
    edges = cv.Canny(image, 100, 200)
    pointconut = area1(edges)
    print('二值图像白色像素个数（表征面积）：', pointconut)

    data = pd.DataFrame([pointconut])
    data = data.T
    data.to_csv('Cannycount_web_steam_augmented.csv', mode='a', header=False)
    # data.to_csv('Cannycount_steam_augmented.csv', mode='a', header=False)

#Data augmentation methods
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

        image = cv.imread(imgpath)
        edges = cv.Canny(image, 100, 200)
        pointconut = area1(edges)
        print('二值图像白色像素个数（表征面积）：', pointconut)

        data = pd.DataFrame([pointconut])
        data = data.T
        data.to_csv('Cannycount_web_steam_augmented.csv', mode='a', header=False)
        # data.to_csv('Cannycount_steam_augmented.csv', mode='a', header=False)