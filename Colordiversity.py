import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd


# 直方图均衡化
def equalHist_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dst = cv2.equalizeHist(gray)
    # cv2.equalizeHist(img)，将要均衡化的原图像【要求是灰度图像】作为参数传入，则返回值即为均衡化后的图像
    cv2.imshow('equalHist_demo', dst)


# CLAHE 图像增强方法主要用在医学图像上面，增强图像的对比度的同时可以抑制噪声，是一种对比度受限情况下的自适应直方图均衡化算法
# 图像对比度指的是一幅图片中最亮的白和最暗的黑之间的反差大小。
def clahe_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    dst = clahe.apply(gray)
    cv2.imshow('clahe_demo', dst)


def create_rgb_hist(image):
    # 创建RGB三通道直方图（直方图矩阵）
    # 16*16*16的意思是三通道的每通道有16个bins
    h, w, c = image.shape
    rgbHist = np.zeros([16 * 16 * 16, 1], np.float32)
    bsize = 256 / 16
    for row in range(h):
        for col in range(w):
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            # 人为构建直方图矩阵的索引，该索引是通过每一个像素点的三通道值进行构建
            index = np.int(b / bsize) * 16 * 16 + np.int(g / bsize) + np.int(r / bsize)
            # 该处形成的矩阵即为直方图矩阵
            rgbHist[np.int(index), 0] = rgbHist[np.int(index), 0] + 1
    plt.ylim([0, 10000])
    plt.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.3)

    return rgbHist


# 直方图比较
def hist_compare(image1, image2):
    hist1 = create_rgb_hist(image1)
    hist2 = create_rgb_hist(image2)
    match1 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    match2 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    match3 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    print('巴氏距离:%s,相关性:%s,卡方:%s' % (match1, match2, match3))
    return match1, match2, match3


# 巴氏距离比较（method=cv2.HISTCMP_BHATTACHARYYA)值越小，相关度越高[0,1]
# 相关性（method=cv2.HISTCMP_CORREL）值越大，相关度越高，[0,1]
# 卡方（method=cv2.HISTCMP_CHISQR）,值越小，相关度越高，[0,inf)


if __name__ == '__main__':
    # image=cv2.imread('../opencv-python-img/lena.png')
    # cv2.imshow('origin_image',image)
    # equalHist_demo(image)
    # clahe_demo(image)
    image2 = cv2.imread(
        'D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/avatar/uniform.jpg')
    for i in range(len(data)):
        imgpath = 'D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/web/web' + str(
            i + 1) + '.jpg'
        # imgpath = 'D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/avatar/avatar' + str(i + 1) + '.jpg'
        img = Image.open(imgpath)
        if img.mode == 'P':  # 必须是RGB模式 P是GIF的格式
            img = img.convert('RGB')
            img.save(
                'D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/web/web' + str(
                    i + 1) + '.jpg')
        # img.save('D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/avatar/avatar' + str(i + 1) + '.jpg')
        src = cv2.imread(imgpath)
        match1, match2, match3 = hist_compare(src, image2)
        data = pd.DataFrame([match1, match2, match3])
        data = data.T
        data.to_csv('Colordiversity_web_steam_augmented.csv', mode='a', header=False)
    # data.to_csv('Colordiversity_steam_augmented.csv', mode='a', header=False)

    text = ['blur', 'brightness']

    for m in text:
        for i in range(len(data)):
            imgpath = 'D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/web/web' + m + str(
                i + 1) + '.jpg'
            # imgpath = 'D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/avatar/avatar' + m + str(i + 1) + '.jpg'
            img = Image.open(imgpath)
            if img.mode == 'P':  # 必须是RGB模式 P是GIF的格式
                img = img.convert('RGB')
                img.save(
                    'D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/web/web' + m + str(
                        i + 1) + '.jpg')
            # 'D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/avatar/avatar' + m + str(i + 1) + '.jpg')

            src = cv2.imread(imgpath)
            match1, match2, match3 = hist_compare(src, image2)
            data = pd.DataFrame([match1, match2, match3])
            data = data.T
            data.to_csv('Colordiversity_web_steam_augmented.csv', mode='a', header=False)
        # data.to_csv('Colordiversity_steam_augmented.csv', mode='a', header=False)

# plt.subplot(1,2,1)
# plt.title('diff1')
# plt.plot(create_rgb_hist(image1))
# plt.subplot(1,2,2)
# plt.title('diff2')
# plt.plot(create_rgb_hist(image2))
# plt.show()

# cv2.waitKey(0)
