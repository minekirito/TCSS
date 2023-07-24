#!/usr/bin/env python
# coding: utf-8
import cv2

#读取图像
def getImage(img_path):
    image = cv2.imread(img_path)
    return image
 
#显示图像  
def disImage(image):
    cv2.imshow("demo",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#获取HSV空间
def get_hsv(image):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    hsv=cv2.resize(hsv,(600,600))
    return hsv

#从HSV获取图像直方图
import numpy as np

def getHistog(image):
    histSize = 256
    histRange = (0, 256)
    accumulate = False
    bgr_planes=cv2.split(image)
    
    b_hist = cv2.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
    g_hist = cv2.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
    r_hist = cv2.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)
    
    assert(sum(b_hist) == image.shape[0] *image.shape[1])
    return b_hist,g_hist,r_hist
    
    
#可以删掉
def getHImage(b_hist,g_hist,r_hist):
    hist_w = 512
    hist_h = 400
    histSize = 256
    bin_w = int(round(hist_w/histSize))
    histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    cv2.normalize(b_hist, b_hist, alpha=0, beta=hist_h,
                 norm_type=cv2.NORM_MINMAX)
    cv2.normalize(g_hist, g_hist, alpha=0, beta=hist_h,
                 norm_type=cv2.NORM_MINMAX)
    cv2.normalize(r_hist, r_hist, alpha=0, beta=hist_h,
                 norm_type=cv2.NORM_MINMAX)
    for i in range(1, histSize):
        cv2.line(histImage, (bin_w*(i-1), hist_h - int(np.round(b_hist[i-1]))),
                (bin_w*(i), hist_h - int(np.round(b_hist[i]))),
                (255, 0, 0), thickness=2)
        cv2.line(histImage, (bin_w*(i-1), hist_h - int(np.round(g_hist[i-1]))),
                (bin_w*(i), hist_h - int(np.round(g_hist[i]))),
                (0, 255, 0), thickness=2)
        cv2.line(histImage, (bin_w*(i-1), hist_h - int(np.round(r_hist[i-1]))),
                (bin_w*(i), hist_h - int(np.round(r_hist[i]))),
                (0, 0, 255), thickness=2)
    cv2.imshow('dst image', histImage)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    return 0


import os
import fnmatch
#获取平均颜色直方图
def getUniformHistog(im_path):
    res=[]
    for folderName,subFolders,fileNames in os.walk(im_path):
        for filename in fileNames:
            if fnmatch.fnmatch(filename,"*.jpg") :
                # res.append(os.path.join(os.getcwd(),filename))
                res.append(os.path.join('D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/web',filename))

    num = len(res)
    bs=0
    gs=0
    rs=0

    for e in res:
        print(e)
        imgage = getImage(e)
        b1,g1,r1 = getHistog(imgage)
        bs = bs+b1
        gs = gs+g1
        rs = rs+r1

    be = bs/num
    ge = gs/num
    re = rs/num
    return be,ge,re,res


def hist2signature(hist):
        signature = np.zeros(shape=(hist.shape[0] * hist.shape[1], 2), dtype=np.float32)
        for h in range(hist.shape[0]):
            idx = h
            signature[idx][0] = hist[h]
            signature[idx][1] = h
        return signature


def EMD_com(b1,g1,r1,be,ge,re):
    signature1 = hist2signature(b1)  
    signature2 = hist2signature(g1)
    signature3 = hist2signature(r1)
    signaturebe = hist2signature(be)  
    signaturege = hist2signature(ge)
    signaturere = hist2signature(re)
    retvalEMD12, lowerBound12, flow12 = cv2.EMD(signature1, signaturebe, cv2.DIST_L2)
    retvalEMD13, lowerBound13, flow13 = cv2.EMD(signature2, signaturege, cv2.DIST_L2)
    retvalEMD14, lowerBound14, flow14 = cv2.EMD(signature3, signaturere, cv2.DIST_L2)
    return retvalEMD12+retvalEMD13+retvalEMD14


import pandas as pd
if __name__=="__main__":
    test_path = r"D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/web"
    b,g,r,res = getUniformHistog(test_path)
    EMD = []
    for e in res:
        image = getImage(e)
        hsv = get_hsv(image)
        hsv = get_hsv(image)
        bb,gg,rr = getHistog(hsv)
        e_emd = EMD_com(b,g,r,bb,gg,rr)
        EMD.append(e_emd)
        print(e_emd)
    data = pd.DataFrame(EMD)
    data = data.T
    data.to_csv('EMD_web_steam_augmented.csv', mode='a', header=False)