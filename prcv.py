import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
import torch
from fastai import *
from fastai.vision import *
import warnings
from albumentations import CLAHE,GaussianBlur,RandomGamma

def img_enhance(img):
    gauss = GaussianBlur()
    img = gauss.apply(img,ksize=5)
    #高斯模糊
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    clahe = CLAHE()
    img = clahe.apply(img)
    #对比度受限的自适应直方图均衡化
    s = img.shape
    v = 0
    for i in range(s[0]):
        for j in range(s[1]):
            v += hsv[i][j][2]
    v = v / s[0] / s[1]
    if v > 120:
        gamma = RandomGamma()
        img = gamma.apply(img,gamma=1.2)
    elif v < 80:
        gamma = RandomGamma()
        img = gamma.apply(img, gamma=0.8)
    #如果图像过亮或过暗，则进行亮伽马变换

    return img
#图像增强

def save_img_data(file,new_file,is_flip):
    dirs = os.listdir(file)
    for i in range(len(dirs)):
        _img = cv.imread(file+dirs[i],cv.IMREAD_COLOR)
        _img = cv.cvtColor(_img,cv.COLOR_BGR2RGB)
        _img = img_enhance(_img)
        _img = cv.resize(_img, dsize=(224, 224))
        img_name = new_file+dirs[i]
        _img = cv.cvtColor(_img,cv.COLOR_RGB2BGR)
        cv.imwrite(img_name,_img)
        if is_flip:
            cv.imwrite(img_name[:-4]+'_flip'+'.jpg',cv.flip(_img,1))
        #存储训练集和预测集的图像翻转后的图像
#数据读取与转存

def save_features(file):
    f = open(file+'labels.txt','r')
    labels1 = open('train/labels1.csv','a')#任务1的标签文件
    labels2 = open('train/labels2.csv', 'a')  # 任务1的标签文件
    labels3 = open('train/labels3.csv','a')#任务3的标签文件
    for line in f:
        line = line[:-1]
        d = line.split(' ')
        name = d[0]
        dr_label = d[1]
        focus = d[2:]
        labels1.write('images/'+name+'.jpg,'+dr_label[-1]+'\n')
        labels1.write('images/' + name + '_flip.jpg,' + dr_label[-1] + '\n')

        t = '0'
        if int(dr_label[-1]) <= 1:
            t = '0'
        else:
            t = '1'
        labels2.write('images/'+name+'.jpg,'+t+'\n')
        labels2.write('images/' + name + '_flip.jpg,' + t + '\n')

        a = ''
        if focus[0] == '':
            a = ',no_illness'
        else:
            a = ','
            for b in focus:
                a = a + b + ' '
            a = a[:-1]
        labels3.write('images/' + name + '.jpg')
        labels3.write(a+'\n')
        labels3.write('images/' + name + '_flip.jpg')
        labels3.write(a + '\n')
#标签读取与转存

def takeSecond(elem):
    return elem[1]
def save_pred_data(learn,task,plan):
    files = os.listdir('test/images/')
    result = open('2017202021/'+task+'_2017202021_'+'run'+plan+'.txt','w')
    ill = ['cotton_wool_spots', 'fibrous_proliferation', 'hard_exudate', 'microaneurysm', 'neovascularization','no_illness',
         'preretinal_hemorrhage', 'retinal_hemorrhage', 'vitreous_hemorrhage']
    for f in files:
        img_name = 'test/images/' + f
        img = open_image(img_name)
        if task == 'task1':
            result.write(f[:-4] + ' ' + str(learn.predict(img)[0]) + '\n')
        elif task == 'task2':
            t = learn.predict(img)
            c = t[0]
            p = t[2].numpy()
            sum = p[0] + p[1]
            p[0] = p[0] / sum
            p[1] = p[1] / sum
            result.write(f[:-4] + ' ' +str(c) + ' ' + str(p[int(c)]) + '\n')
        else:
            result.write(f[:-4])
            p = learn.predict(img)[2].numpy()
            illness = str(learn.predict(img)[0]).split(';')
            if illness[0] == '':
                result.write('\n')
            #每个标签的概率都小于0.5时认为没有疾病
            else:
                s = []
                for i in range(9):
                    s.append((ill[i], p[i]))
                s.sort(key=takeSecond, reverse=True)
                if s[0][0] == 'no_illness':
                    result.write('\n')
                    continue
                #当无疾病的概率最大时，认为没有疾病
                for d in s:
                    if d[1] > 0.5 and d[0] != 'no_illness':
                        result.write(' ' + d[0])
                result.write('\n')
            #存在标签的概率大于0.5时，输出疾病信息
#对测试集进行预测并存储结果

def F1score(y_pred,y_true):
    n = y_true.shape[0]
    y_true = y_pred.argmax(dim=-1).view(n, -1)
    return fbeta(y_pred,y_true,beta=1)
#利用fbeta求F1score

if __name__ == '__main__':
    save_img_data('data/dr_train/images_896x896/','train/images/',True)
    save_img_data('data/dr_val/images_896x896/','train/images/',True)
    save_img_data('data/dr_test/images_896x896/','test/images/',False)
    # 将图像进行增强和翻转并保存
    labels1 = open('train/labels1.csv', 'w')
    labels2 = open('train/labels2.csv', 'w')
    labels3 = open('train/labels3.csv', 'w')
    labels1.write('name,label\n')
    labels2.write('name,label\n')
    labels3.write('name,label\n')
    labels1.close()
    labels2.close()
    labels3.close()
    save_features('data/dr_train/')
    save_features('data/dr_val/')
    #存储图像的标签信息

    warnings.filterwarnings("ignore")
    kappa = KappaScore()
    kappa.weights = "quadratic"#加权kappa
    tfms = get_transforms(do_flip=False, flip_vert=True, max_rotate=360, max_warp=0, max_zoom=1.1, max_lighting=0.1,
                          p_lighting=0.75, p_affine=0.75)
    #随机对输入的图像进行旋转、亮度增强等操作

    #task1
    df = pd.read_csv('train/labels1.csv')
    data1 = (ImageList.from_df(df=df, path='train', cols='name')
            .split_by_rand_pct(0.12)
            .label_from_df(cols="label")
            .transform(tfms, size=224, resize_method=ResizeMethod.SQUISH, padding_mode='zeros')
            .databunch(bs=16, num_workers=0)
            .normalize(imagenet_stats)
            )
    #读入训练集和预测集的数据，并从所有图像中选取10%作为预测集
    learn = cnn_learner(data1, base_arch=models.squeezenet1_0, metrics=[accuracy, kappa])#使用fastai的cnn学习器，模型使用squeezenet1_0，训练中反馈信息为accuracy和加权kappa
    lr_find(learn,1e-7,1e-1,num_it=100)
    learn.recorder.plot(suggestion=True)
    plt.show()
    #寻找合适的学习率
    x = input("input the best learning_rate:\n")
    learn.fit(100, lr=float(x))#模型学习
    y = input("want to save this result? y/n\n")
    if y == 'y':
        save_pred_data(learn=learn,task='task1',plan='1')

    learn1 = cnn_learner(data1, base_arch=models.squeezenet1_1, metrics=[accuracy, kappa])#使用fastai的cnn学习器，模型使用squeezenet1_1，训练中反馈信息为accuracy和加权kappa
    lr_find(learn1, 1e-7, 1e-1, num_it=100)
    learn1.recorder.plot(suggestion=True)
    plt.show()
    # 寻找合适的学习率
    x = input("input the best learning_rate:\n")
    learn1.fit(50, lr=float(x))  # 模型学习
    y = input("want to save this result? y/n\n")
    if y == 'y':
        save_pred_data(learn=learn1, task='task1', plan='2')

    learn2 = cnn_learner(data1, base_arch=models.vgg16_bn,
                        metrics=[accuracy, kappa])  # 使用fastai的cnn学习器，模型使用vgg16_bn，此时batch_size=16，训练中反馈信息为accuracy和加权kappa
    lr_find(learn2, 1e-7, 1e-1, num_it=100)
    learn2.recorder.plot(suggestion=True)
    plt.show()
    # 寻找合适的学习率
    x = input("input the best learning_rate:\n")
    learn2.fit(40, lr=float(x))  # 模型学习
    y = input("want to save this result? y/n\n")
    if y == 'y':
        save_pred_data(learn=learn2, task='task1', plan='3')

    #task2
    df = pd.read_csv('train/labels2.csv')
    data2 = (ImageList.from_df(df=df, path='train', cols='name')
             .split_by_rand_pct(0.12)
             .label_from_df(cols="label")
             .transform(tfms, size=224, resize_method=ResizeMethod.SQUISH, padding_mode='zeros')
             .databunch(bs=64, num_workers=0)
             .normalize(imagenet_stats)
             )
    learn = cnn_learner(data2,base_arch=models.squeezenet1_0,metrics=[accuracy,AUROC(),F1score])#使用fastai的cnn学习器，模型使用squeezenet1_0，训练中反馈信息为accuracy和AUC以及F1socre
    lr_find(learn, 1e-7, 1e-1, num_it=100)
    learn.recorder.plot(suggestion=True)
    plt.show()
    # 寻找合适的学习率
    x = input("input the best learning_rate:\n")
    learn.fit(30, lr=float(x))  # 模型学习
    y = input("want to save this result? y/n\n")
    if y == 'y':
        save_pred_data(learn=learn, task='task2', plan='1')
    preds,y,losses = learn.get_preds(with_loss=True)
    interp = ClassificationInterpretation(learn, preds, y, losses)
    cm = interp.confusion_matrix()#混淆矩阵

    learn1 = cnn_learner(data2, base_arch=models.squeezenet1_1,
                        metrics=[accuracy,AUROC(),F1score])  # 使用fastai的cnn学习器，模型使用squeezenet1_1，训练中反馈信息为accuracy和AUC以及F1socre

    lr_find(learn1, 1e-7, 1e-1, num_it=100)
    learn1.recorder.plot(suggestion=True)
    plt.show()
    # 寻找合适的学习率
    x = input("input the best learning_rate:\n")
    learn1.fit(30, lr=float(x))  # 模型学习
    y = input("want to save this result? y/n\n")
    if y == 'y':
        save_pred_data(learn=learn1, task='task2', plan='2')
    preds,y,losses = learn1.get_preds(with_loss=True)
    interp = ClassificationInterpretation(learn1, preds, y, losses)
    cm = interp.confusion_matrix()#混淆矩阵
    print(cm)


    df = pd.read_csv('train/labels3.csv')
    data3 = (ImageList.from_df(df=df, path='train', cols='name')
             .split_by_rand_pct(0.12)
             .label_from_df(cols="label",label_delim=' ')
             .transform(tfms, size=224, resize_method=ResizeMethod.SQUISH, padding_mode='zeros')
             .databunch(bs=64, num_workers=0)
             .normalize(imagenet_stats)
             )
    learn = cnn_learner(data3, base_arch=models.squeezenet1_0, metrics=[MultiLabelFbeta()])  # 使用fastai的cnn学习器，模型使用squeezenet1_0，训练中反馈信息为多标签的Fbeta值
    lr_find(learn, 1e-7, 1e-1, num_it=100)
    learn.recorder.plot(suggestion=True)
    plt.show()
    # 寻找合适的学习率
    x = input("input the best learning_rate:\n")
    learn.fit(30, lr=float(x))  # 模型学习
    y = input("want to save this result? y/n\n")
    if y == 'y':
        save_pred_data(learn=learn, task='task3', plan='1')

    learn2 = cnn_learner(data3, base_arch=models.squeezenet1_1,
                        metrics=[MultiLabelFbeta()])  # 使用fastai的cnn学习器，模型使用squeezenet1_1，训练中反馈信息为多标签的Fbeta值
    lr_find(learn2, 1e-7, 1e-1, num_it=100)
    learn2.recorder.plot(suggestion=True)
    plt.show()
    # 寻找合适的学习率
    x = input("input the best learning_rate:\n")
    learn2.fit(30, lr=float(x))  # 模型学习
    y = input("want to save this result? y/n\n")
    if y == 'y':
        save_pred_data(learn=learn2, task='task3', plan='2')