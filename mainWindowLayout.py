#!/usr/bin/python3
# -*- coding: utf-8 -*-

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from PyQt5 import QtCore, QtGui, QtWidgets
import cv2


class MainLayout(object):

    def setupUi(self, window):
        # 设置窗口大小
        window.resize(1000, 800)
        # 设置背景色
        self.setObjectName("MainWindow")
        self.setStyleSheet("#MainWindow{background-color: #19232D}")
        # 窗口标题
        self.setWindowTitle('SyImageApp')
        self.setWindowIcon(QIcon('./Logo.ico'))
        self.centralWidget = QtWidgets.QWidget(window)
        # 全局布局
        mainLayout = QVBoxLayout(self.centralWidget)
        self.label = QLabel()
        self.show()

        # 顶部布局
        # 顶部固定布局
        topLayout = QVBoxLayout()
        solidLayout = QHBoxLayout()   # 功能按钮1
        solidLayout2 = QVBoxLayout()  # 路径显示
        solidLayout3 = QHBoxLayout()  # 功能按钮2
        self.importImageEdit = QLineEdit('请点击下方“文件”菜单选择需要处理的文件!   PowerBy KK')
        self.importImageEdit.setFocusPolicy(Qt.NoFocus)
        solidLayout2.addWidget(self.importImageEdit)

        # test
        # self.testButton = QPushButton('test')
        # self.testButton.clicked.connect(self.on_click())
        # solidLayout.addWidget(self.testButton)

        # 文件按钮
        self.importButton = QPushButton('文件')
        
        solidLayout.addWidget(self.importButton)
        filemenu = QMenu(self)
        self.openAct = QAction('打开', self)
        filemenu.addAction(self.openAct)
        # self.saveAct = QAction('保存',self)
        # filemenu.addAction(self.saveAct)
        self.exitAct = QAction('退出', self)
        filemenu.addAction(self.exitAct)
        self.importButton.setMenu(filemenu)

        # 编辑按钮
        self.editButton = QPushButton('坐标变换')
        solidLayout.addWidget(self.editButton)
        editmenu = QMenu(self)
        self.shiftAct = QAction('平移变换', self)
        editmenu.addAction(self.shiftAct)
        self.largeAct = QAction('放大', self)
        editmenu.addAction(self.largeAct)
        self.smallAct = QAction('缩小', self)
        editmenu.addAction(self.smallAct)
        self.rotateAct = QAction('旋转', self)
        editmenu.addAction(self.rotateAct)
        self.affineAct = QAction('仿射变换', self)
        editmenu.addAction(self.affineAct)
        self.grayAct = QAction('灰度', self)
        editmenu.addAction(self.grayAct)
        self.brightAct = QAction('亮度', self)
        editmenu.addAction(self.brightAct)
        self.screenshotAct = QAction('截图', self)
        editmenu.addAction(self.screenshotAct)
        self.editButton.setMenu(editmenu)

        # 灰度映射按钮
        self.grayButton = QPushButton('灰度映射')
        solidLayout.addWidget(self.grayButton)
        graymenu = QMenu(self)
        # 求反
        self.revAct = QAction('求反', self)
        graymenu.addAction(self.revAct)
        # 动态范围压缩
        self.logAct = QAction('动态范围压缩', self)
        graymenu.addAction(self.logAct)
        # 阶梯量化
        self.stepAct = QAction('阶梯量化', self)
        graymenu.addAction(self.stepAct)
        # 阈值分割
        self.thresholdAct = QAction('阈值分割', self)
        graymenu.addAction(self.thresholdAct)
        self.grayButton.setMenu(graymenu)

        # 算术运算
        self.arithmeticButton = QPushButton('算术运算')
        solidLayout.addWidget(self.arithmeticButton)
        arithmeticmenu = QMenu(self)
        # 图像加法
        self.addAct = QAction('图像加法', self)
        arithmeticmenu.addAction(self.addAct)
        # 中值去噪
        self.medianAct = QAction('中值去噪', self)
        arithmeticmenu.addAction(self.medianAct)
        # 图像减法
        self.subAct = QAction('图像减法', self)
        arithmeticmenu.addAction(self.subAct)
        self.arithmeticButton.setMenu(arithmeticmenu)
        
        # 直方图修正
        # 直方图均衡化
        self.equcorButton = QPushButton('直方图修正')
        solidLayout.addWidget(self.equcorButton)
        equcormenu = QMenu(self)
        self.equ1Act = QAction('直方图均衡化', self)
        equcormenu.addAction(self.equ1Act)
        # 直方图规定化
        self.equ2Act = QAction('直方图规定化', self)
        equcormenu.addAction(self.equ2Act)
        self.equcorButton.setMenu(equcormenu)
        
        # 空域滤波
        self.spatialButton = QPushButton('空域滤波')
        solidLayout.addWidget(self.spatialButton)
        spatialmenu = QMenu(self)
        # 线性平滑滤波
        self.spatial1Act = QAction('线性平滑滤波', self)
        spatialmenu.addAction(self.spatial1Act)
        # 线性锐化滤波
        self.spatial2Act = QAction('线性锐化滤波', self)
        spatialmenu.addAction(self.spatial2Act)
        # 非线性平滑滤波
        self.spatial3Act = QAction('非线性平滑滤波', self)
        spatialmenu.addAction(self.spatial3Act)
        # 非线性锐化滤波
        self.spatial4Act = QAction('非线性锐化滤波', self)
        spatialmenu.addAction(self.spatial4Act)
        self.spatialButton.setMenu(spatialmenu)
        
        # 傅里叶变换
        self.fourierButton = QPushButton('傅里叶变换')
        solidLayout.addWidget(self.fourierButton)
        fouriermenu = QMenu(self)
        # 傅里叶变换self.fourier1Act
        self.fourier1Act = QAction('傅里叶变换', self)
        fouriermenu.addAction(self.fourier1Act)
        # 傅里叶逆变换
        self.fourier2Act = QAction('傅里叶逆变换', self)
        fouriermenu.addAction(self.fourier2Act)
        self.fourierButton.setMenu(fouriermenu)
        
        # 高通和低通滤波器
        self.gdlbButton = QPushButton('高通和低通滤波器')
        solidLayout.addWidget(self.gdlbButton)
        gdlbmenu = QMenu(self)
        # 高通滤波器self.gdlb1Act
        self.gdlb1Act = QAction('高通滤波器', self)
        gdlbmenu.addAction(self.gdlb1Act)
        # 低通滤波器
        self.gdlb2Act = QAction('低通滤波器', self)
        gdlbmenu.addAction(self.gdlb2Act)
        self.gdlbButton.setMenu(gdlbmenu)
        
        # 变换按钮
        self.changeButton = QPushButton('变换')
        solidLayout.addWidget(self.changeButton)
        changemenu = QMenu(self)
        self.change1Act = QAction('傅里叶变换', self)
        changemenu.addAction(self.change1Act)
        self.change2Act = QAction('离散余弦变换', self)
        changemenu.addAction(self.change2Act)
        self.change3Act = QAction('Radon变换', self)
        changemenu.addAction(self.change3Act)
        self.changeButton.setMenu(changemenu)
        # 噪声按钮
        self.noiseButton = QPushButton('噪声')
        solidLayout.addWidget(self.noiseButton)
        noisemenu = QMenu(self)
        self.noise1Act = QAction('高斯噪声', self)
        noisemenu.addAction(self.noise1Act)
        self.noise2Act = QAction('椒盐噪声', self)
        noisemenu.addAction(self.noise2Act)
        self.noise3Act = QAction('斑点噪声', self)
        noisemenu.addAction(self.noise3Act)
        self.noise4Act = QAction('泊松噪声', self)
        noisemenu.addAction(self.noise4Act)
        self.noiseButton.setMenu(noisemenu)
        # 滤波按钮
        self.smoothingButton = QPushButton('滤波')
        solidLayout.addWidget(self.smoothingButton)
        smoothingmenu = QMenu(self)
        self.smoothing1Act = QAction('高通滤波', self)
        smoothingmenu.addAction(self.smoothing1Act)
        self.smoothing2Act = QAction('低通滤波', self)
        smoothingmenu.addAction(self.smoothing2Act)
        self.smoothing3Act = QAction('平滑滤波', self)
        smoothingmenu.addAction(self.smoothing3Act)
        self.smoothing4Act = QAction('锐化滤波', self)
        smoothingmenu.addAction(self.smoothing4Act)
        self.smoothingButton.setMenu(smoothingmenu)
        # 直方图统计按钮
        self.histButton = QPushButton('直方图统计')
        solidLayout.addWidget(self.histButton)
        histmenu = QMenu(self)
        self.hist1Act = QAction('R直方图', self)
        histmenu.addAction(self.hist1Act)
        self.hist2Act = QAction('G直方图', self)
        histmenu.addAction(self.hist2Act)
        self.hist3Act = QAction('B直方图', self)
        histmenu.addAction(self.hist3Act)
        self.histButton.setMenu(histmenu)
        # 图像增强按钮
        self.enhanceButton = QPushButton('图像增强')
        solidLayout.addWidget(self.enhanceButton)
        enhancemenu = QMenu(self)
        self.enhance1Act = QAction('伪彩色增强', self)
        enhancemenu.addAction(self.enhance1Act)
        self.enhance2Act = QAction('真彩色增强', self)
        enhancemenu.addAction(self.enhance2Act)
        self.enhance3Act = QAction('直方图均衡', self)
        enhancemenu.addAction(self.enhance3Act)
        self.enhance4Act = QAction('NTSC颜色模型', self)
        enhancemenu.addAction(self.enhance4Act)
        self.enhance5Act = QAction('YCbCr颜色模型', self)
        enhancemenu.addAction(self.enhance5Act)
        self.enhance6Act = QAction('HSV颜色模型', self)
        enhancemenu.addAction(self.enhance6Act)
        self.enhanceButton.setMenu(enhancemenu)
        # 阈值分割按钮
        self.threButton = QPushButton('阈值分割')
        solidLayout.addWidget(self.threButton)
        # 形态学处理按钮
        self.morphologyProcessButton = QPushButton('形态学处理')
        solidLayout.addWidget(self.morphologyProcessButton)
        # 特征提取按钮
        self.featureButton = QPushButton('特征提取')
        solidLayout.addWidget(self.featureButton)
        # 图像分类与识别按钮
        self.imgButton = QPushButton('图像分类与识别')
        solidLayout.addWidget(self.imgButton)
        solidLayout.addStretch(1)
        topLayout.addLayout(solidLayout2)   # 显示文件路径
        topLayout.addLayout(solidLayout)    # 显示功能按钮1
        topLayout.addLayout(solidLayout3)   # 显示功能按钮2

        # 顶部隐藏布局
        self.hideLayout = QHBoxLayout()
        topLayout.addLayout(self.hideLayout)
        mainLayout.addLayout(topLayout)

        # 中间布局
        midLayout = QHBoxLayout()
        self.showImageView = QTableWidget()
        midLayout.addWidget(self.showImageView)
        mainLayout.addLayout(midLayout)

        # 底部布局
        # bottomLayout=QHBoxLayout()
        # self.preButton=QPushButton('上一张')
        # bottomLayout.addWidget(self.preButton)
        # self.nextButton=QPushButton('下一张')
        # bottomLayout.addWidget(self.nextButton)
        # bottomLayout.addStretch(4)
        # self.exitButton=QPushButton('退出')
        # bottomLayout.addWidget(self.exitButton)
        # mainLayout.addLayout(bottomLayout)

        # 设置stretch
        mainLayout.setStretchFactor(topLayout, 1)
        mainLayout.setStretchFactor(midLayout, 6)
        # mainLayout.setStretchFactor(bottomLayout,1)

        window.setCentralWidget(self.centralWidget)
        