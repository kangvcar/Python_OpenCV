import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from pywt import dwt2, idwt2
# 定义需要处理的图像
pic1 = './images/image1.jpg'
pic2 = './images/image2.jpg'
pic3 = './images/image3.jpg'


def imgShift(pic):
    """
    图像平移是将一副图像中所有的点都按照指定的平移量在水平、
    垂直方向移动，平移后的图像与原图像相同。

    Args:
        pic (str): 图像路径

    Return:
        平移后的图像
    """
    image = cv.imread(pic)
    rows, cols = image.shape[:2]
    # 设置偏移量
    M = np.float32([[1, 0, 100], [0, 1, 50]])
    # 仿射变换
    dst = cv.warpAffine(image, M, (cols, rows))
    ax1 = plt.subplot(1, 2, 1)
    plt.imshow(image)
    ax1.set_title("source")
    ax2 = plt.subplot(1, 2, 2)
    plt.imshow(dst)
    ax2.set_title("deal")
    plt.savefig('./result/' + 'imgshift' + pic, format='jpg')


def imgResize(pic):
    """
    空间尺度变换是为了在一系列的空间尺度上提取一副图像的空间信息，从而得到从小区域的细节得到图
    像中大的特征信息。这些算法类似于过滤器，被重复地用于不同空间尺度上，或者过滤器本身被尺度化，
    可以把它们归于一系列空间尺寸过滤器。这类过滤器逐渐受到较高的重视，是因为它们将遥感图像的空
    间信息从局部到整体表现在一系列不同的空间尺度上，从而提供了一种表达图像信息的方法。

    Args:
        pic (str): 图像路径

    Return:
        尺度变换后的图像
    """
    image = cv.imread(pic)
    height, width = image.shape[:2]
    # 在X轴和Y轴上进行0.5倍的缩放，采用像素区域关系重新采样的插值方法
    dst = cv.resize(image, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
    # 使用matplotlib来进行绘图，同时将原图像和处理后图像一起呈现
    ax1 = plt.subplot(1, 2, 1)
    plt.imshow(image)
    ax1.set_title("原图")
    ax2 = plt.subplot(1, 2, 2)
    plt.imshow(dst)
    ax2.set_title("尺度变换后")
    plt.savefig('./result/' + 'imgresize' + pic, format='jpg')


def imgRotate(pic):
    """
    一般图像的旋转是以图像的中心为原点，旋转一定角度，即将图像上的所有像素都旋转一个相同的角度。

    Args:
        pic (str): 图像路径

    Return:
        旋转变换后的图像
    """
    image = cv.imread(pic)
    rows, cols = image.shape[:2]
    # 设置旋转中心，旋转角度，以及旋转后的缩放比例
    M = cv.getRotationMatrix2D(((cols - 1) / 2, (rows - 1) / 2), 45, 1)
    # 仿射变换
    dst = cv.warpAffine(image, M, (rows, cols), borderValue=(255, 255, 255))
    ax1 = plt.subplot(1, 2, 1)
    plt.imshow(image)
    ax1.set_title("原图")
    ax2 = plt.subplot(1, 2, 2)
    plt.imshow(dst)
    ax2.set_title("旋转变换后")
    plt.savefig('./result/' + 'imgrotate' + pic, format='jpg')


def imgAffine(pic):
    """
    仿射变换是在几何上定义为两个向量之间的有一个仿射变换或者仿射映射，由一个非奇异的线性变换
    (运用一次函数进行的变换)接上一个平移变换组成。在有限维的情况下，每个仿射变换可以由一个
    矩阵A和一个向量b组成。一个仿射变换对应于一个矩阵和一个向量的乘法，而仿射变换的复合对应于
    普通的矩阵乘法，只要加入一个额外的行到矩阵的底下，这一行全部是0除了最右边是有一个1，而
    列向量的底下要加一个1。在仿射变换中，原始图像中的所有平行线在输出图像中仍然是平行的。
    为了找到变换矩阵，我们需要从输入图像中取三个点及其在输出图像中的对应位置。
    然后 cv.getAffineTransform 将创建一个 2x3 矩阵，该矩阵将传递给 cv.warpAffine。

    Args:
        pic (str): 图像路径

    Return:
        仿射变换后的图像
    """
    image = cv.imread(pic)
    rows, cols = image.shape[:2]
    # 设置三个变换前后的的位置对应点
    pos1 = np.float32([[50, 50], [300, 50], [50, 200]])
    pos2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv.getAffineTransform(pos1, pos2)  # 设置变换矩阵M
    dst = cv.warpAffine(image, M, (rows, cols))  # 仿射变换
    ax1 = plt.subplot(1, 2, 1)
    plt.imshow(image)
    ax1.set_title("原图")
    ax2 = plt.subplot(1, 2, 2)
    plt.imshow(dst)
    ax2.set_title("仿射变换后")
    plt.savefig('./result/' + 'imgaffine' + pic, format='jpg')


def grayscaleMapping(pic):
    """
    灰度变换是指根据某种目标条件按一定变换关系逐点改变原图像中每一个像素灰度值的方法。目的是为
    了改善画质，使得图像的显示效果更加清晰。图像的灰度变换处理是图像增强处理技术的一种非常基
    础、直接的空间域图像处理方法，也是图像数字化软件和图像显示软件的一个重要组成部分。阈值分割
    法是一种基于区域的图像分割技术，原理是把图像像素分为若干类。图像阈值化分割是一种传统的最常
    用的图像分割技术，因为其实现简单、计算量小、性能稳定而成为图像分割中最基本和应用最广泛的分
    割技术。他特别适用于目标和背景占据不同灰度级范围的图像。它不仅可以极大的压缩数据量，而且也
    大大简化了分析和处理步骤，因此在很多情况下，是进行图像分析、特征处理与模式识别之前的必要的
    图像预处理过程。图像阈值化的目的是要按照灰度级，对像素集合进行一个划分，得到的每个子集形成
    一个与现实景物相对应的区域，每个区域内部具有一致的属性，而相邻区域不具有这种一致属性。这样
    的划分可以通过从灰度级出发选取一个或者多个阈值来实现。

    Args:
        pic (str): 图像路径

    Return:
        灰度映射后的图像
    """
    image = cv.imread(pic, 0)  # 获取灰度值
    rev_img = 255 - np.array(image)  # 取反
    log_img = np.uint8(42 * np.log(1.0 + image))  # 动态范围压缩
    step_img = np.zeros((image.shape[0], image.shape[1]))  # 阶梯量化
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (image[i, j] <= 230) and (image[i, j] >= 120):
                step_img[i, j] = 0
            else:
                step_img[i, j] = image[i, j]
    threshold_img = cv.adaptiveThreshold(image, 254,
                                         cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv.THRESH_BINARY, 11, 2)  # 阈值分割
    ax1 = plt.subplot(2, 2, 1)
    plt.imshow(rev_img, cmap='gray')
    ax1.set_title("取反")
    ax2 = plt.subplot(2, 2, 2)
    plt.imshow(log_img, cmap='gray')
    ax2.set_title("动态范围压缩")
    ax3 = plt.subplot(2, 2, 3)
    plt.imshow(step_img, cmap='gray')
    ax3.set_title("阶梯量化")
    ax4 = plt.subplot(2, 2, 4)
    plt.imshow(threshold_img, cmap='gray')
    ax4.set_title("阈值分割")
    plt.savefig('./result/' + 'grayscalemapping' + pic, format='jpg')


def arithmeticOperation(pic):
    """
    图像的代数运算是指对两幅或两幅以上的输入图像的对应元素逐个进行和、差、积、商的四则运算，以
    产生有增强效果的图像。图像代数运算是一种比较简单和有效的增强处理，是遥感图像增强处理中常用
    的一种方法。

    Args:
        pic (str): 图像路径

    Return:
        算术运算后的图像
    """
    image = cv.imread(pic)
    add_img = cv.addWeighted(image, 0.8, image, 0.5, 0)  # 相加
    img_medianBlur = cv.medianBlur(image, 3)  # 中值滤波
    sub_img = image - image  # 相减
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(image)
    ax[0, 1].imshow(add_img)
    ax[1, 0].imshow(img_medianBlur)
    ax[1, 1].imshow(sub_img)
    ax[0, 0].set_title("原图")
    ax[0, 1].set_title("图像相加")
    ax[1, 0].set_title("中值滤波")
    ax[1, 1].set_title("图像相减")
    fig.tight_layout()
    plt.savefig('./result/' + 'arithmeticOperation' + pic, format='jpg')