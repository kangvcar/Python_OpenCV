#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import os
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from mainWindowLayout import MainLayout

import cv2
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt


class MainWindow(QMainWindow, MainLayout):
    imagePaths = []
    originImages = []
    imageList = []  # 二维的图像列表
    hideLayoutTag = -1

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.signalSlots()

    # button与具体方法关联
    def signalSlots(self):
        # 文件按钮相关方法
        # 打开
        self.openAct.triggered.connect(lambda: importImage(self))
        # 保存
        # self.saveAct.triggered.connect(lambda : importImage(self))
        # 退出
        self.exitAct.triggered.connect(self.close)

        # 编辑按钮相关方法
        # 放大
        self.shiftAct.triggered.connect(lambda: shiftImage(self))
        # 放大
        self.largeAct.triggered.connect(lambda: largeImage(self))
        # 缩小
        self.smallAct.triggered.connect(lambda: smallImage(self))
        # 灰度
        self.grayAct.triggered.connect(lambda: grayImage(self))
        # 亮度
        self.brightAct.triggered.connect(lambda: brightImage(self))
        # 旋转
        self.rotateAct.triggered.connect(lambda: rotateImage(self))
        # 仿射变换
        self.affineAct.triggered.connect(lambda: affineImage(self))
        # 截图
        self.screenshotAct.triggered.connect(lambda: screenshotImage(self))

        # 灰度映射按钮相关方法
        # 求反
        self.revAct.triggered.connect(lambda: revImage(self))
        # 动态范围压缩
        self.logAct.triggered.connect(lambda: logImage(self))
        # 阶梯量化
        self.stepAct.triggered.connect(lambda: stepImage(self))
        # 阈值分割
        self.thresholdAct.triggered.connect(lambda: thresholdImage(self))
        # 图像加法
        self.addAct.triggered.connect(lambda: addImage(self))
        # 中值去噪
        self.medianAct.triggered.connect(lambda: medianImage(self))
        # 图像减法
        self.subAct.triggered.connect(lambda: subImage(self))
        # 直方图均衡化
        self.equ1Act.triggered.connect(lambda: equ1Image(self))
        # 直方图规定化
        self.equ2Act.triggered.connect(lambda: equ2Image(self))
        # 线性平滑滤波
        self.spatial1Act.triggered.connect(lambda: spatial1Image(self))
        # 线性锐化滤波
        self.spatial2Act.triggered.connect(lambda: spatial2Image(self))
        # 非线性平滑滤波
        self.spatial3Act.triggered.connect(lambda: spatial3Image(self))
        # 非线性锐化滤波
        self.spatial4Act.triggered.connect(lambda: spatial4Image(self))
        # 傅里叶变换
        self.fourier1Act.triggered.connect(lambda: fourier1Image(self))
        # 傅里叶逆变换
        self.fourier2Act.triggered.connect(lambda: fourier2Image(self))
        # 高通滤波器
        self.gdlb1Act.triggered.connect(lambda: smoothing1Image(self))
        # 高通滤波器
        self.gdlb2Act.triggered.connect(lambda: smoothing2Image(self))

        # template --> fourier1Image
        # def OIMAGE(window):
        #     imageList = []
        #     for img in window.originImages:
        #         imgs = []
        #         pass
        #         imgs.extend([img[0], result])
        #         imageList.append(imgs)
        #         resizeFromList(window, imageList)
        #         showImage(window, ['原图', ''])

        # 变换按钮相关方法
        # 傅里叶变换
        self.change1Act.triggered.connect(lambda: change1Image(self))
        # 离散余弦变换
        self.change2Act.triggered.connect(lambda: change2Image(self))
        # Radon变换
        self.change3Act.triggered.connect(lambda: change3Image(self))

        # 噪声按钮相关方法
        # 高斯噪声
        self.noise1Act.triggered.connect(lambda: noise1Image(self))
        # 椒盐噪声
        self.noise2Act.triggered.connect(lambda: noise2Image(self))
        # 斑点噪声
        self.noise3Act.triggered.connect(lambda: importImage(self))
        # 泊松噪声
        self.noise4Act.triggered.connect(lambda: importImage(self))

        # 滤波按钮相关方法
        # 高通滤波
        self.smoothing1Act.triggered.connect(lambda: smoothing1Image(self))
        # 低通滤波
        self.smoothing2Act.triggered.connect(lambda: smoothing2Image(self))
        # 平滑滤波
        self.smoothing3Act.triggered.connect(lambda: smoothing3Image(self))
        # 锐化滤波
        self.smoothing4Act.triggered.connect(lambda: smoothing4Image(self))

        # 直方图统计按钮相关方法
        # R直方图
        self.hist1Act.triggered.connect(lambda: hist1Image(self))
        # G直方图
        self.hist2Act.triggered.connect(lambda: importImage(self))
        # B直方图
        self.hist3Act.triggered.connect(lambda: importImage(self))

        # 图像增强按钮相关方法
        # 伪彩色增强
        self.enhance1Act.triggered.connect(lambda: enhance1Image(self))
        # 真彩色增强
        self.enhance2Act.triggered.connect(lambda: enhance2Image(self))
        # 直方图均衡
        self.enhance3Act.triggered.connect(lambda: histNormalized(self))
        # NTSC颜色模型
        self.enhance4Act.triggered.connect(lambda: enhance4Image(self))
        # YCbCr颜色模型
        self.enhance5Act.triggered.connect(lambda: enhance5Image(self))
        # HSV颜色模型
        self.enhance6Act.triggered.connect(lambda: enhance6Image(self))

        # 阈值分割方法
        self.threButton.clicked.connect(lambda: threImage(self))
        # 形态学处理方法
        self.morphologyProcessButton.clicked.connect(
            lambda: morphologyProcessImage(self))
        # 特征提取方法
        self.featureButton.clicked.connect(lambda: featureImage(self))
        # 图像分类与识别方法
        self.imgButton.clicked.connect(lambda: layoutChange(self))
        # 底部
        # #上一张
        # self.preButton.clicked.connect(lambda : preImage(self))
        # #下一张
        # self.nextButton.clicked.connect(lambda : nextImage(self))
        # 退出
        # self.exitButton.clicked.connect(self.close)


# 坐标变换按钮相关方法
# 平移变换


def shiftImage(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        rows, cols = img[0].shape[:2]
        M = np.float32([[1, 0, 100], [0, 1, 50]])
        result = cv2.warpAffine(img[0], M, (cols, rows))
        cv2.imshow("move", result)
        imgs.extend([img[0], result])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window, ['原图', '平移后'])


# 放大


def largeImage(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        result = cv2.resize(img[0],
                            None,
                            fx=2,
                            fy=2,
                            interpolation=cv2.INTER_CUBIC)
        cv2.imshow("large", result)
        imgs.extend([img[0], result])
        imageList.append(imgs)
    # resizeFromList(window, imageList)
    showImage(window, ['原图', '放大后'])


# 缩小


def smallImage(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        result = cv2.resize(img[0],
                            None,
                            fx=0.5,
                            fy=0.5,
                            interpolation=cv2.INTER_CUBIC)
        cv2.imshow("small", result)
        imgs.extend([img[0], result])
        imageList.append(imgs)
    # resizeFromList(window, imageList)
    showImage(window, ['原图', '缩小后'])


# 灰度


def grayImage(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        result = cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB)
        imgs.extend([img[0], result])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window, ['原图', '灰度处理后'])


# 亮度


def brightImage(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        rows, cols, chunnel = img[0].shape
        blank = np.zeros([rows, cols, chunnel], img[0].dtype)
        result = cv2.addWeighted(img[0], 1.3, blank, 1 - 1.3, 3)
        imgs.extend([img[0], result])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window, ['原图', '调整亮度后'])


# 旋转


def rotateImage(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        rows, cols = img[0].shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
        result = cv2.warpAffine(img[0],
                                M, (rows, cols),
                                borderValue=(255, 255, 255))
        imgs.extend([img[0], result])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window, ['原图', '旋转后'])


# 仿射变换


def affineImage(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        rows, cols = img[0].shape[:2]
        pos1 = np.float32([[50, 50], [300, 50], [50, 200]])
        pos2 = np.float32([[10, 100], [200, 50], [100, 250]])
        M = cv2.getAffineTransform(pos1, pos2)
        result = cv2.warpAffine(img[0], M, (rows, cols))
        imgs.extend([img[0], result])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window, ['原图', '仿射变换后'])


# 截图


def screenshotImage(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        result = img[0][70:170, 440:540]
        imgs.extend([img[0], result])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window, ['原图', '截图后'])


# 灰度映射按钮相关方法
# 求反


def revImage(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        result = 255 - np.array(img[0])
        imgs.extend([img[0], result])
        imageList.append(imgs)
        resizeFromList(window, imageList)
        showImage(window, ['原图', '求反后'])


# 动态范围压缩


def logImage(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        result = np.uint8(42 * np.log(1.0 + img[0]))
        imgs.extend([img[0], result])
        imageList.append(imgs)
        resizeFromList(window, imageList)
        showImage(window, ['原图', '动态范围压缩后'])


# 阶梯量化


def stepImage(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        result = np.zeros((img[0].shape[0], img[0].shape[1]))
        for i in range(img[0].shape[0]):
            for j in range(img[0].shape[1]):
                # print('img[0][i, j]=', img[0][i, j], type(img[0][i, j]))
                # if (img[0][i, j] <= [230,230,230]) and (img[0][i, j] >= [120,120,120]):
                if np.logical_and(img[0][i, j] < 230, img[0][i, j] > 120):
                    result[i, j] = 0
                else:
                    result[i, j] = img[0][i, j]
        imgs.extend([img[0], result])
        imageList.append(imgs)
        cv2.imshow(result)
    resizeFromList(window, imageList)
    showImage(window, ['原图', '阶梯量化后'])


# 阈值分割


def thresholdImage(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        result = cv2.adaptiveThreshold(img[0], 254,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        imgs.extend([img[0], result])
        imageList.append(imgs)
        cv2.imshow(result)
        resizeFromList(window, imageList)
        showImage(window, ['原图', '阈值分割后'])


# 加法


def addImage(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        result = cv2.add(img[0], img[0])
        imgs.extend([img[0], result])
        imageList.append(imgs)
        resizeFromList(window, imageList)
        showImage(window, ['原图', '加法后'])


# 平均值去噪


def medianImage(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        result = cv2.medianBlur(img[0], 3)
        imgs.extend([img[0], result])
        imageList.append(imgs)
        resizeFromList(window, imageList)
        showImage(window, ['原图', '平均值去噪后'])


# 减法


def subImage(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        result = img[0] - img[0]
        imgs.extend([img[0], result])
        imageList.append(imgs)
        resizeFromList(window, imageList)
        showImage(window, ['原图', '减法后'])


# 直方图均衡化


def equ1Image(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        img[0] = cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY)
        result = cv2.equalizeHist(img[0])
        imgs.extend([img[0], result])
        imageList.append(imgs)
        resizeFromList(window, imageList)
        showImage(window, ['原图', '直方图均衡化后'])


# 直方图规定化


def equ2Image(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        result = np.zeros_like(img[0])
        _, colorChannel = img[0].shape[:2]
        for i in range(colorChannel):
            hist_img, _ = np.histogram(img[0][:, i], 256)
            hist_ref, _ = np.histogram(img[0][:, i], 256)
            cdf_img = np.cumsum(hist_img)
            cdf_ref = np.cumsum(hist_ref)
            for j in range(256):
                tmp = abs(cdf_img[j] - cdf_ref)
                tmp = tmp.tolist()
                idx = tmp.index(min(tmp))
                result[:, i][img[0][:, i] == j] = idx

        imgs.extend([img[0], result])
        imageList.append(imgs)
        resizeFromList(window, imageList)
        showImage(window, ['原图', '直方图规定化后'])


# 线性平滑滤波


def spatial1Image(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        result = cv2.blur(img[0], (7, 7))
        imgs.extend([img[0], result])
        imageList.append(imgs)
        resizeFromList(window, imageList)
        showImage(window, ['原图', '线性平滑滤波后'])


# 线性锐化滤波


def spatial2Image(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        kernel_sharpen_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        result = cv2.filter2D(img[0], -1, kernel_sharpen_1)
        imgs.extend([img[0], result])
        imageList.append(imgs)
        resizeFromList(window, imageList)
        showImage(window, ['原图', '线性锐化滤波后'])


# 非线性平滑滤波


def spatial3Image(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        result = cv2.medianBlur(img[0], 5)
        imgs.extend([img[0], result])
        imageList.append(imgs)
        resizeFromList(window, imageList)
        showImage(window, ['原图', '非线性平滑滤波后'])


# 非线性锐化滤波


def spatial4Image(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        result = cv2.bilateralFilter(img[0], 5, 31, 31)
        imgs.extend([img[0], result])
        imageList.append(imgs)
        resizeFromList(window, imageList)
        showImage(window, ['原图', '非线性锐化滤波后'])


# 傅里叶变换


def fourier1Image(window):
    # 读取图像
    img = cv2.imread(window.imagePaths[0], 0)
    # 傅里叶变换
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)  # 傅里叶变换库函数调用
    dftshift = np.fft.fftshift(dft)  # 将傅里叶频域从左上角移动到中间
    # 双通道结果转换为0到255的范围用于图像显示
    res1 = 20 * np.log(cv2.magnitude(dftshift[:, :, 0], dftshift[:, :, 1]))
    # 傅里叶逆变换
    ishift = np.fft.ifftshift(dftshift)  # 将频域从中间移动到左上角
    iimg = cv2.idft(ishift)  # 傅里叶逆变换库函数调用
    res2 = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])  # 双通道结果转换为0到255的范围
    # 显示图像
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    plt.subplot(131), plt.imshow(img, 'gray'), plt.title('原图像')
    plt.axis('off')
    plt.subplot(132), plt.imshow(res1, 'gray'), plt.title('傅里叶变换')
    plt.axis('off')
    # plt.subplot(133), plt.imshow(res2, 'gray'), plt.title('傅里叶逆变换')
    # plt.axis('off')
    plt.show()


# 傅里叶逆变换


def fourier2Image(window):
    # 读取图像
    img = cv2.imread(window.imagePaths[0], 0)
    # 傅里叶变换
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)  # 傅里叶变换库函数调用
    dftshift = np.fft.fftshift(dft)  # 将傅里叶频域从左上角移动到中间
    # 双通道结果转换为0到255的范围用于图像显示
    res1 = 20 * np.log(cv2.magnitude(dftshift[:, :, 0], dftshift[:, :, 1]))
    # 傅里叶逆变换
    ishift = np.fft.ifftshift(dftshift)  # 将频域从中间移动到左上角
    iimg = cv2.idft(ishift)  # 傅里叶逆变换库函数调用
    res2 = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])  # 双通道结果转换为0到255的范围
    # 显示图像
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    plt.subplot(131), plt.imshow(img, 'gray'), plt.title('原图像')
    plt.axis('off')
    # plt.subplot(132), plt.imshow(res1, 'gray'), plt.title('傅里叶变换')
    # plt.axis('off')
    plt.subplot(132), plt.imshow(res2, 'gray'), plt.title('傅里叶逆变换')
    plt.axis('off')
    plt.show()


# 变换按钮相关方法
# 傅里叶变换


def change1Image(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        b, g, r = cv2.split(img[0])
        b_freImg, b_recImg = oneChannelDft(b)
        g_freImg, g_recImg = oneChannelDft(g)
        r_freImg, r_recImg = oneChannelDft(r)
        freImg = cv2.merge([b_freImg, g_freImg, r_freImg])
        imgs.extend([img[0], freImg])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window, ['原图', '傅里叶变换后'])


def oneChannelDft(img):
    width, height = img.shape
    nwidth = cv2.getOptimalDFTSize(width)
    nheigth = cv2.getOptimalDFTSize(height)
    nimg = np.zeros((nwidth, nheigth))
    nimg[:width, :height] = img
    dft = cv2.dft(np.float32(nimg), flags=cv2.DFT_COMPLEX_OUTPUT)
    ndft = dft[:width, :height]
    ndshift = np.fft.fftshift(ndft)
    magnitude = np.log(cv2.magnitude(ndshift[:, :, 0], ndshift[:, :, 1]))
    result = (magnitude - magnitude.min()) / \
        (magnitude.max() - magnitude.min()) * 255
    frequencyImg = result.astype('uint8')
    ilmg = cv2.idft(dft)
    ilmg = cv2.magnitude(ilmg[:, :, 0], ilmg[:, :, 1])[:width, :height]
    ilmg = np.floor((ilmg - ilmg.min()) / (ilmg.max() - ilmg.min()) * 255)
    recoveredImg = ilmg.astype('uint8')
    return frequencyImg, recoveredImg


# 离散余弦变换


def change2Image(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        img1 = cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB)
        img_dct = cv2.dct(img1)  # 进行离散余弦变换
        imgs.extend([img[0], img_dct])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window, ['原图', '离散余弦变换后'])


# Radon变换


def change3Image(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        img_dct = cv2.dct(img[0])
        result = np.log(abs(img_dct))
        imgs.extend([img[0], result])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window, ['原图', 'Radon变换后'])


# 噪声按钮相关方法
# 高斯噪声
# 定义添加高斯噪声的函数


def addGaussianNoise(image, percetage):
    G_Noiseimg = image
    G_NoiseNum = int(percetage * image.shape[0] * image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(20, 40)
        temp_y = np.random.randint(20, 40)
        G_Noiseimg[temp_x][temp_y] = 255
    return G_Noiseimg


def noise1Image(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        grayImage = cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB)  # 灰度变换
        result = addGaussianNoise(grayImage, 0.01)  # 添加10%的高斯噪声
        imgs.extend([img[0], result])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window, ['原图', '高斯噪声后'])


# 椒盐噪声
# 定义添加椒盐噪声的函数
def saltpepper(img, n):
    m = int((img.shape[0] * img.shape[1]) * n)
    for a in range(m):
        i = int(np.random.random() * img.shape[1])
        j = int(np.random.random() * img.shape[0])
        if img.ndim == 2:
            img[j, i] = 255
        elif img.ndim == 3:
            img[j, i, 0] = 255
            img[j, i, 1] = 255
            img[j, i, 2] = 255
    for b in range(m):
        i = int(np.random.random() * img.shape[1])
        j = int(np.random.random() * img.shape[0])
        if img.ndim == 2:
            img[j, i] = 0
        elif img.ndim == 3:
            img[j, i, 0] = 0
            img[j, i, 1] = 0
            img[j, i, 2] = 0
    return img


def noise2Image(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        grayImage = cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB)  # 灰度变换
        result = saltpepper(grayImage, 0.02)
        imgs.extend([img[0], result])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window, ['原图', '椒盐噪声后'])


# 滤波按钮相关方法
# 高通滤波
def smoothing1Image(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        x = cv2.Sobel(img[0], cv2.CV_16S, 1, 0)
        y = cv2.Sobel(img[0], cv2.CV_16S, 0, 1)
        absx = cv2.convertScaleAbs(x)
        absy = cv2.convertScaleAbs(y)
        result = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
        imgs.extend([img[0], result])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window, ['原图', '高通滤波后'])


# 低通滤波
def smoothing2Image(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        result = cv2.medianBlur(img[0], 5)
        imgs.extend([img[0], result])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window, ['原图', '低通滤波后'])


# 平滑滤波
def smoothing3Image(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        result = cv2.blur(img[0], (5, 5))
        imgs.extend([img[0], result])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window, ['原图', '平滑滤波后'])


# 锐化滤波
def smoothing4Image(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        result = cv2.bilateralFilter(img[0], 9, 75, 75)
        imgs.extend([img[0], result])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window, ['原图', '锐化滤波后'])


# 直方图统计按钮相关方法
# R直方图
def hist1Image(window):
    imageList = []
    for img in window.originImages:
        imgs = []
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([img[0]], [i], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        plt.savefig("hist1.jpg")

        result = cv2.imread("hist1.jpg")
        imgs.extend([img[0], result])
        imageList.append(imgs)
    resizeFromList(window, imageList)
    showImage(window, ['原图', 'R直方图后'])


# 图像增强按钮相关方法
# 伪彩色增强
def enhance1Image(window):
    imageList = []

    for img in window.originImages:
        imgs = []
        grayImage = cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB)  # 灰度变换
        result = cv2.applyColorMap(grayImage, cv2.COLORMAP_JET)
        imgs.extend([img[0], result])
        imageList.append(imgs)

    resizeFromList(window, imageList)
    showImage(window, ['原图', '伪彩色增强后'])


# 真彩色增强
def enhance2Image(window):
    imageList = []

    for img in window.originImages:
        imgs = []
        grayImage = cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB)  # 灰度变换
        result = cv2.applyColorMap(grayImage, cv2.COLORMAP_JET)
        imgs.extend([img[0], result])
        imageList.append(imgs)

    resizeFromList(window, imageList)
    showImage(window, ['原图', '真彩色增强后'])


# 直方图均衡
def histNormalized(window):
    imageList = []

    for img in window.originImages:
        imgs = []
        b, g, r = cv2.split(img[0])
        b_equal = cv2.equalizeHist(b)
        g_equal = cv2.equalizeHist(g)
        r_equal = cv2.equalizeHist(r)
        result = cv2.merge([b_equal, g_equal, r_equal])
        imgs.extend([img[0], result])
        imageList.append(imgs)

    resizeFromList(window, imageList)
    showImage(window, ['原图', '直方图均衡化后'])


# NTSC颜色模型
def enhance4Image(window):
    imageList = []

    for img in window.originImages:
        imgs = []
        result = cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB)
        imgs.extend([img[0], result])
        imageList.append(imgs)

    resizeFromList(window, imageList)
    showImage(window, ['原图', 'NTSC颜色模型后'])


# YCbCr颜色模型
def enhance5Image(window):
    imageList = []

    for img in window.originImages:
        imgs = []
        result = cv2.cvtColor(img[0], cv2.COLOR_BGR2YCR_CB)
        imgs.extend([img[0], result])
        imageList.append(imgs)

    resizeFromList(window, imageList)
    showImage(window, ['原图', 'YCbCr颜色模型后'])


# HSV颜色模型
def enhance6Image(window):
    imageList = []

    for img in window.originImages:
        imgs = []
        result = cv2.cvtColor(img[0], cv2.COLOR_BGR2HSV)
        imgs.extend([img[0], result])
        imageList.append(imgs)

    resizeFromList(window, imageList)
    showImage(window, ['原图', 'HSV颜色模型后'])


# 阈值分割方法
def threImage(window):
    imageList = []

    for img in window.originImages:
        imgs = []
        grayImage = cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB)  # 灰度变换
        result = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
        imgs.extend([img[0], result])
        imageList.append(imgs)

    # resizeFromList(window, imageList)
    showImage(window, ['原图', '阈值分割后'])


# 形态学处理方法
def morphologyProcessImage(window):
    imageList = []

    for img in window.originImages:
        imgs = []
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        result = cv2.erode(img[0], kernel)
        imgs.extend([img[0], result])
        imageList.append(imgs)

    resizeFromList(window, imageList)
    showImage(window, ['原图', '形态学处理后'])


# 特征提取方法
def featureImage(window):
    imageList = []

    for img in window.originImages:
        imgs = []
        img1 = img[0].copy()
        gray = cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        img[0][dst > 0.01 * dst.max()] = [0, 0, 255]
        imgs.extend([img1, img[0]])
        imageList.append(imgs)

    resizeFromList(window, imageList)
    showImage(window, ['原图', '特征提取后'])


# 打开图像
def importImage(window):
    fname, _ = QFileDialog.getOpenFileName(
        window, 'Open file', '.',
        'Image Files(*.jpg *.bmp *.png *.jpeg *.rgb *.tif)')
    if fname != '':
        window.importImageEdit.setText(fname)
        window.imagePaths = []
        window.originImages = []
        window.imageList = []
        window.imagePaths.append(fname)
    if window.imagePaths != []:
        readIamge(window)
        resizeFromList(window, window.originImages)
        showImage(window)


def readIamge(window):
    window.originImages = []
    for path in window.imagePaths:
        imgs = []
        # img=cv2.imread(path)
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
        # img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        imgs.append(img)
        window.originImages.append(imgs)


# 显示图像
def showImage(window, headers=[]):
    window.showImageView.clear()
    window.showImageView.setColumnCount(len(window.imageList[0]))
    window.showImageView.setRowCount(len(window.imageList))

    window.showImageView.setShowGrid(False)
    window.showImageView.setEditTriggers(QAbstractItemView.NoEditTriggers)
    window.showImageView.setHorizontalHeaderLabels(headers)
    for x in range(len(window.imageList[0])):
        for y in range(len(window.imageList)):
            imageView = QGraphicsView()
            imageView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            imageView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

            img = window.imageList[y][x]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            width = img.shape[1]
            height = img.shape[0]

            window.showImageView.setColumnWidth(x, width)
            window.showImageView.setRowHeight(y, height)

            frame = QImage(img, width, height, QImage.Format_RGB888)
            # 调用QPixmap命令，建立一个图像存放框
            pix = QPixmap.fromImage(frame)
            item = QGraphicsPixmapItem(pix)
            scene = QGraphicsScene()  # 创建场景
            scene.addItem(item)
            imageView.setScene(scene)
            window.showImageView.setCellWidget(y, x, imageView)


def resizeFromList(window, imageList):
    width = 600
    height = 600
    window.imageList = []
    for x_pos in range(len(imageList)):
        imgs = []
        for img in imageList[x_pos]:
            # image=cv2.resize(img, (width, height))
            image = cv2.resize(img, (width, height),
                               interpolation=cv2.INTER_CUBIC)
            imgs.append(image)
        window.imageList.append(imgs)
        print(len(window.imageList), len(window.imageList[0]))


def resizeFromBigList(window, imageList):
    width = 1000
    height = 1000
    window.imageList = []
    for x_pos in range(len(imageList)):
        imgs = []
        for img in imageList[x_pos]:
            # image=cv2.resize(img, (width, height))
            image = cv2.resize(img, (width, height),
                               interpolation=cv2.INTER_CUBIC)
            imgs.append(image)
        window.imageList.append(imgs)
        print(len(window.imageList), len(window.imageList[0]))


def resizeFromSmallList(window, imageList):
    width = 300
    height = 300
    window.imageList = []
    for x_pos in range(len(imageList)):
        imgs = []
        for img in imageList[x_pos]:
            # image=cv2.resize(img, (width, height))
            image = cv2.resize(img, (width, height),
                               interpolation=cv2.INTER_CUBIC)
            imgs.append(image)
        window.imageList.append(imgs)
        print(len(window.imageList), len(window.imageList[0]))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())
