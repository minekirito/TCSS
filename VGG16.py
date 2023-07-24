import os
import numpy as np
import pandas as pd
import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable
import torch.cuda
import torchvision.transforms as transforms

from PIL import Image

TARGET_IMG_SIZE = 224
img_to_tensor = transforms.ToTensor()


def make_model():
    model = models.vgg16(pretrained=True) # 其实就是定位到第28层，对照着上面的key看就可以理解
    model.classifier[6]=torch.nn.Sequential(torch.nn.Linear(4096,4096))

    #由于我们只有2个类别需要预测，并且VGG16在ImageNet上有1000个类别，因此我们需要根据任务更新最后一层，因此我们将只训练最后一层，可以通过设置该层中的requires_grad=True来只对最后一层进行权值更新。让我们将训练设置为GPU训练：
    # model = models.vgg16(pretrained=True).features[:28]  # 其实就是定位到第28层，对照着上面的key看就可以理解
    # model = models.vgg16(pretrained=True).classifier.add_module('Linear',torch.nn.Linear(1000,10)) # 其实就是定位到第28层，对照着上面的key看就可以理解
    for param in model.classifier[6].parameters():
        param.requires_grad = True
    model = model.eval()  # 一定要有这行，不然运算速度会变慢（要求梯度）而且会影响结果
    # model.cuda()  # 将模型从CPU发送到GPU,如果没有GPU则删除该行

    return model


# 特征提取
def extract_feature(model, imgpath):
    model.eval()  # 必须要有，不然会影响特征提取结果

    img = Image.open(imgpath)  # 读取图片
    img = img.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))
    img = img.convert("RGB") #不知道为什么 要跑web就要加上这句 
    tensor = img_to_tensor(img)  # 将图片转化成tensor
    print(tensor.shape) # [3, 224, 224]
    # tensor = tensor.cuda()  # 如果只是在cpu上跑的话要将这行去掉
    tensor = Variable(torch.unsqueeze(tensor, dim=0).float(), requires_grad=False)
    print(tensor.shape)  # [1,3, 224, 224]
    result = model(Variable(tensor))
    result_npy = result.data.cpu().numpy()  # 保存的时候一定要记得转成cpu形式的，不然可能会出错

    return result_npy[0]  # 返回的矩阵shape是[1, 512, 14, 14]，这么做是为了让shape变回[512, 14,14]

from PIL import Image

if __name__ == "__main__":
    model = make_model()
    for i in range(len(data)):
        imgpath = 'D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/web/web'+ str(i+1) +'.jpg'
        # imgpath = 'D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/avatar/avatar'+ str(i+1) +'.jpg'
        img = Image.open(imgpath)
        if img.mode == 'P':  #必须是RGB模式 P是GIF的格式
            img = img.convert('RGB')
            img.save('D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/web/web'+ str(i+1) +'.jpg')
            # img.save('D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/avatar/avatar'+ str(i+1) +'.jpg')

        tmp = extract_feature(model, imgpath)
        print(tmp.shape)  # 打印出得到的tensor的shape
        print(tmp)  # 打印出tensor的内容，其实可以换成保存tensor的语句，这里的话就留给读者自由发挥了
        data = pd.DataFrame(tmp)
        data = data.T
        data.to_csv('4096_web_steam_augmented.csv',mode='a',header=False)
        # data.to_csv('4096_steam_augmented.csv',mode='a',header=False)

    text = ['blur', 'brightness']

    for m in text :
        for i in range(len(data)):
            imgpath = 'D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/web/web'+ m + str(i+1) +'.jpg'
            # imgpath = 'D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/avatar/avatar'+ m + str(i+1) +'.jpg'
            img = Image.open(imgpath)
            if img.mode == 'P':  #必须是RGB模式 P是GIF的格式
                img = img.convert('RGB')
                img.save('D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/web/web'+ m + str(i+1) +'.jpg')
                # img.save('D:/Desktop/personality/personality-prediction-master/personality-prediction-master/data/personality_steam/avatar/avatar'+ m + str(i+1) +'.jpg')

            tmp = extract_feature(model, imgpath)
            print(tmp.shape)  # 打印出得到的tensor的shape
            print(tmp)  # 打印出tensor的内容，其实可以换成保存tensor的语句，这里的话就留给读者自由发挥了
            data = pd.DataFrame(tmp)
            data = data.T
            data.to_csv('4096_web_steam_augmented.csv',mode='a',header=False)
            # data.to_csv('4096_steam_augmented.csv',mode='a',header=False)
