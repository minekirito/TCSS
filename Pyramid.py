import cv2 as cv
import numpy as np
from PIL import Image
import pandas as pd

def feature_hog(image):
    winSize = (32, 32)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (16, 16)
    nbins = 5
    hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    test_hog = hog.compute(image, (64, 64), (0, 0)).reshape((-1,))
    # print(test_hog)
    return test_hog

#高斯金字塔
def gaussian_pyramid(image):
    level = 3#设置金字塔的层数为3
    temp = image.copy()  #拷贝图像
    gaussian_images = []  #建立一个空列表
    for i in range(level):
        dst = cv.pyrDown(temp)   #先对图像进行高斯平滑，然后再进行降采样（将图像尺寸行和列方向缩减一半）
        gaussian_images.append(dst)  #在列表末尾添加新的对象
        cv.imshow("gaussian"+str(i), dst)
        temp = dst.copy()
    return gaussian_images


#拉普拉斯金字塔
def laplacian_pyramid(image):
    feature_hog_laplacian = []
    gaussian_images = gaussian_pyramid(image)    #做拉普拉斯金字塔必须用到高斯金字塔的结果
    level = len(gaussian_images)
    for i in range(level-1, -1, -1):
        if (i-1) < 0:
            expand = cv.pyrUp(gaussian_images[i], dstsize = image.shape[:2])
            laplacian = cv.subtract(image, expand)
            # 展示差值图像
            cv.imshow("laplacian_down_"+str(i), laplacian)
            temp_laplacian = feature_hog(laplacian)
            feature_hog_laplacian.extend(temp_laplacian)

        else:
            expand = cv.pyrUp(gaussian_images[i], dstsize = gaussian_images[i-1].shape[:2])
            laplacian = cv.subtract(gaussian_images[i-1], expand)
            # 展示差值图像
            cv.imshow("laplacian_down_"+str(i), laplacian)
            temp_laplacian = feature_hog(laplacian)
            feature_hog_laplacian.extend(temp_laplacian)
    print('all_feature_hog_laplacian',len(feature_hog_laplacian))
    return feature_hog_laplacian

if __name__ == "__main__":
    for i in range(len(data)):
        imgpath = 'D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/web/web' + str(i + 1) + '.jpg'
        # imgpath = 'D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/avatar/avatar' + str(i + 1) + '.jpg'
        img = Image.open(imgpath)
        if img.mode == 'P':  # 必须是RGB模式 P是GIF的格式
            img = img.convert('RGB')
            img.save('D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/web/web' + str(
            # img.save('D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/avatar/avatar' + str(
                        i + 1) + '.jpg')
        src = cv.imread(imgpath)
        input_image = cv.resize(src, (560, 560))
        # 设置为 WINDOW_NORMAL 可以任意缩放
        feature_hog_laplacian = laplacian_pyramid(input_image)
        data = pd.DataFrame(feature_hog_laplacian)
        data = data.T
        # data.to_csv('Phog_steam_augmented.csv', mode='a', header=False)
        data.to_csv('Phog_web_steam_augmented.csv', mode='a', header=False)

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

            src = cv.imread(imgpath)
            input_image = cv.resize(src, (560, 560))
            # 设置为 WINDOW_NORMAL 可以任意缩放
            feature_hog_laplacian = laplacian_pyramid(input_image)
            data = pd.DataFrame(feature_hog_laplacian)
            data = data.T
            data.to_csv('Phog_web_steam_augmented.csv', mode='a', header=False)
            # data.to_csv('Phog_steam_augmented.csv', mode='a', header=False)


# src = cv.imread('D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/avatar/avatar2.jpg',cv.IMREAD_GRAYSCALE)
# input_image = cv.resize(src, (560, 560))
# cv.namedWindow('input_image', cv.WINDOW_AUTOSIZE)
# cv.imshow('input_image', src)
# laplacian_pyramid(src)
