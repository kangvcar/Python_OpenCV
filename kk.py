import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from pywt import dwt2, idwt2
# 定义需要处理的图像
pic1 = 'image1.jpg'
pic2 = 'image2.jpg'
pic3 = 'image3.jpg'


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
    ax1.set_title("原图")
    ax2 = plt.subplot(1, 2, 2)
    plt.imshow(dst)
    ax2.set_title("平移后")
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


def histogramCorrection(pic):
    """
    直方图均衡化又被称为灰度均值化，是指通过某种灰度映射使输入图像转换为在每一个灰度级上都有近
    似相同的像素点数的输出图像(即输出的直方图是均匀的)。在经过均衡化处理后的图像中，像素将占有尽
    可能多的灰度级并且分布均匀，因此，这样的图像将具有较高的对比度和较大的动态范围。类似于下
    图，较亮的图像会把所有像素限制在较高的范围内，使用直方图均衡的方法将直方图拉伸到两端。

    Args:
        pic (str): 图像路径

    Return:
        直方图均衡化后,直方图规定化后的图像
    """
    image = cv.imread(pic, 0)
    equ_img = cv.equalizeHist(image)  # 直方图均衡化
    hist = np.zeros_like(image)  # 直方图规定化
    _, colorChannel = image.shape
    for i in range(colorChannel):
        hist_img, _ = np.histogram(image[:, i], 256)
        hist_ref, _ = np.histogram(image[:, i], 256)
        cdf_img = np.cumsum(hist_img)
        cdf_ref = np.cumsum(hist_ref)

        for j in range(256):
            tmp = abs(cdf_img[j] - cdf_ref)
            tmp = tmp.tolist()
            idx = tmp.index(min(tmp))
            hist[:, i][image[:, i] == j] = idx
    ax1 = plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    ax1.set_title("原图")
    ax2 = plt.subplot(1, 3, 2)
    plt.imshow(equ_img, cmap='gray')
    ax2.set_title("直方图均衡化")
    ax3 = plt.subplot(1, 3, 3)
    plt.imshow(hist, cmap='gray')
    ax3.set_title("直方图规定化")
    plt.savefig('./result/' + 'histogramCorrection' + pic, format='jpg')


def spatialFiltering(pic):
    """
    空域：仅考虑空间位置关系的计算，排除时间维度的影响。在数字图像领域，可以表示为I(x, y)，
    即每幅图像中的像素点的数值仅由位置信息(x, y)决定。空间域的算法只会使用单幅图像本身的信息进行运算。
    对于大多数的空域线性滤波而言，其本质上是一个加权计算的操作，可以归结为
    “掩模计算（Image Mask）+归一化计算（Normalize）+卷积计算（Convolution）”。

    Args:
        pic (str): 图像路径
   
    Return:
        线性平滑滤波器、线性锐化滤波器，非线性平滑滤波器、非线性锐化滤波器处理后的图像
    """
    image = cv.imread(pic)
    ls_img = cv.blur(image, (7, 7))  # 线性平滑滤波
    kernel_sharpen_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    lr_img = cv.filter2D(image, -1, kernel_sharpen_1)  # 线性锐化滤波
    nls_img = cv.medianBlur(image, 5)  # 非线性平滑滤波
    nlr_img = cv.bilateralFilter(image, 5, 31, 31)  # 非线性锐化滤波
    plt.subplots_adjust(left=0.0, bottom=0.0, top=1, right=2)
    ax1 = plt.subplot(1, 5, 1)
    plt.imshow(image)
    ax1.set_title("原图")
    ax2 = plt.subplot(1, 5, 2)
    plt.imshow(ls_img)
    ax2.set_title("线性平滑滤波")
    ax3 = plt.subplot(1, 5, 3)
    plt.imshow(lr_img)
    ax3.set_title("线性锐化滤波")
    ax4 = plt.subplot(1, 5, 4)
    plt.imshow(nls_img)
    ax4.set_title("非线性平滑滤波")
    ax5 = plt.subplot(1, 5, 5)
    plt.imshow(nlr_img)
    ax5.set_title("非线性锐化滤波")
    plt.savefig('./result/' + 'spatialFiltering' + pic, format='jpg')


def fourier(pic):
    """
    在数字图像处理中，有两个经典的变换被广泛应用——傅里叶变换和霍夫变化。
    其中，傅里叶变换主要是将时间域上的信号转变为频率域上的信号，用来进行图像降噪，图像增强等处理。
    对于数字图像这种离散的信号，频率大小表示信号变换的剧烈程度或者说信号变化的快慢。
    频率越大，变换越剧烈，频率越小，信号越平缓，对应到的图像中，高频信号往往是图像中的边缘信号和噪声信号，
    而低频信号包含图像变化频繁的图像轮廓及背景灯信号。

    傅里叶变换（Fourier Transform，简称FT）常用于数字信号处理，它的目的是将时间域上的信号转变为频率域上的信号。
    随着域的不同，对同一个事物的了解角度也随之改变，因此在时域某些不好处理的地方，在频域就可以较为简单的处理。
    同时，可以从频域里发现一些原先不易察觉的特征。

    傅里叶定理指出“任何连续周期信号都可以表示成（或者无限逼近）一系列正弦信号的叠加”。

    Args:
        pic (str): 图像路径
   
    Return:
        傅里叶变换和傅里叶逆变换处理后的图像
    """
    image = cv.imread(pic)
    graypic = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # 傅里叶变换
    fo = np.fft.fft2(graypic)
    fshift = np.fft.fftshift(fo)
    fo_img = np.log(np.abs(fshift))

    # 傅里叶逆变换
    f1shift = np.fft.ifftshift(fshift)
    nfo_img = np.fft.ifft2(f1shift)
    nfo_img = np.abs(nfo_img)

    ax1 = plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    ax1.set_title("原图")
    ax2 = plt.subplot(1, 3, 2)
    plt.imshow(fo_img, cmap='gray')
    ax2.set_title("傅里叶变换")
    ax3 = plt.subplot(1, 3, 3)
    plt.imshow(nfo_img, cmap='gray')
    ax3.set_title("傅里叶逆变换")
    plt.savefig('./result/' + 'fourier' + pic, format='jpg')


def HighALowFilter(pic, D0=30, W=None, N=2, type='lhp', filter='ideal'):
    """
    图像的频率滤波是基于傅里叶变换的，通过二维傅里叶变换把图像从空域转换到频域，对频域的图像的
    频率进行操作，比如限制某个频率范围的像素通过。

    理想滤波器：理想低通滤波器在以原点为圆心、D0为半径的园内，通过所有的频率，而在圆外截断所有的频率。
    （圆心的频率最低，为变换的直流(dc)分量）。

    巴特沃斯滤波器：的特点是通频带（passband）内的频率响应曲线最大限度平坦，没有涟波，而在阻频带则逐渐下降为零。

    指数滤波器：衰减快，高频成分少（图像处理时，比巴特沃斯滤波器稍微模糊一点），有平滑的过渡带（无振铃效应）。

    Args:
        pic (str): 图像路径
        D0 (int, optional): 截止频率. Defaults to 30.
        W ([type], optional): 带宽. Defaults to None.
        N (int, optional):  butterworth和指数滤波器的阶数. Defaults to 2.
        type (str, optional): lhp, bp, bs即低通和高通、带通、带阻. Defaults to 'lhp'.
        filter (str, optional): butterworth、ideal、exponential即巴特沃斯、理想、指数滤波器. Defaults to 'ideal'.

    Return:
        高通和低通滤波器处理后的图像
    """
    image = cv.imread(pic, 0)
    dft = cv.dft(np.float32(image), flags=cv.DFT_COMPLEX_OUTPUT)  # 离散傅里叶变换
    dtf_shift = np.fft.fftshift(dft)  # 中心化
    rows, cols = image.shape[0], image.shape[1]
    crow, ccol = int(rows / 2), int(cols / 2)  # 计算频谱中心
    mask = np.ones((rows, cols, 2))  # 生成rows行cols列的2维矩阵
    maskk = np.ones((rows, cols, 2))  # 生成rows行cols列的2维矩阵

    for i in range(rows):
        for j in range(cols):
            D = np.sqrt((i - crow)**2 + (j - ccol)**2)
            if (filter.lower() == 'butterworth'):  # 巴特沃滤波器
                if (type == 'lhp'):
                    mask[i, j] = 1 / (1 + (D / D0)**(2 * N))
                    maskk[i, j] = 1 / (1 + (D0 / D)**(2 * N))
                elif (type == 'bsp'):
                    mask[i, j] = 1 / (1 + (D * W / (D**2 - D0**2))**(2 * N))
                    maskk[i, j] = 1 / (1 + ((D**2 - D0**2) / D * W)**(2 * N))
                else:
                    assert ('type error')

            elif (filter.lower() == 'ideal'):  # 理想滤波器
                if (type == 'lhp'):
                    if (D > D0):
                        mask[i, j] = 0
                    if (D < D0):
                        maskk[i, j]
                elif (type == 'bsp'):
                    if (D > D0 and D < D0 + W):
                        mask[i, j] = 0
                    if (D < D0 and D > D0 + W):
                        maskk[i, j] = 0
                else:
                    assert ('type error')

            elif (filter.lower() == 'exponential'):  # 指数滤波器
                if (type == 'lhp'):
                    mask[i, j] = np.exp(-(D / D0)**(2 * N))
                    maskk[i, j] = np.exp(-(D0 / D)**(2 * N))
                elif (type == 'bsp'):
                    mask[i, j] = np.exp(-(D * W / (D**2 - D0**2))**(2 * N))
                    maskk[i, j] = np.exp(-((D**2 - D0**2) / D * W)**(2 * N))
                else:
                    assert ('type error')

    lfshift = dtf_shift * mask
    hfshift = dtf_shift * maskk

    # 傅里叶逆变换-低通
    lf_ishift = np.fft.ifftshift(lfshift)
    limg_back = cv.idft(lf_ishift)
    limg_back = cv.magnitude(limg_back[:, :, 0], limg_back[:, :,
                                                           1])  # 计算像素梯度的绝对值
    limg_back = np.abs(limg_back)
    limg_back = (limg_back - np.amin(limg_back)) / (np.amax(limg_back) -
                                                    np.amin(limg_back))

    # 傅里叶逆变换-高通
    hf_ishift = np.fft.ifftshift(hfshift)
    himg_back = cv.idft(hf_ishift)
    himg_back = cv.magnitude(himg_back[:, :, 0], himg_back[:, :,
                                                           1])  # 计算像素梯度的绝对值
    himg_back = np.abs(himg_back)
    himg_back = (himg_back - np.amin(himg_back)) / (np.amax(himg_back) -
                                                    np.amin(himg_back))

    ax1 = plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    ax1.set_title("原图")
    ax2 = plt.subplot(1, 3, 2)
    plt.imshow(limg_back, cmap='gray')
    ax2.set_title("低通滤波器")
    ax3 = plt.subplot(1, 3, 3)
    plt.imshow(himg_back, cmap='gray')
    ax3.set_title("高通滤波器")
    plt.savefig('./result/' + 'HighALowFilter' + pic, format='jpg')


def decreaseArray(image1, image2):
    """
    矩阵乘法

    Args:
        image1 (str): 图像路径
        image2 (str): 图像路径

    Returns:
        矩阵乘法处理后图像
    """
    image = image1.copy()
    for i in range(image1.shape[0] - 1):
        for j in range(image1.shape[1] - 1):
            image[i][j] = image1[i][j] - image2[i][j]
            j = j + 1
        i = i + 1
    return image


def increaseArray(image1, image2):
    """
    矩阵加法

    Args:
        image1 (str): 图像路径
        image2 (str): 图像路径

    Returns:
        矩阵加法处理后图像
    """
    image = image1.copy()
    for i in range(image1.shape[0] - 1):
        for j in range(image1.shape[1] - 1):
            image[i][j] = image1[i][j] + image2[i][j]
            j = j + 1
        i = i + 1
    return image


def hight_pass_filter(pic):
    """
    特殊高通滤波器

    Args:
        pic (str): 图像路径

    Return:
        特殊高通滤波器处理后图像
    """
    image = cv.imread(pic)
    imageAver3 = cv.blur(image, (3, 3))
    upsharpMask = decreaseArray(image, imageAver3)
    imageSharp = increaseArray(image, upsharpMask)

    ax1 = plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    ax1.set_title("原图")
    ax2 = plt.subplot(2, 2, 2)
    plt.imshow(imageAver3, cmap='gray')
    ax2.set_title("模糊图像")
    ax3 = plt.subplot(2, 2, 3)
    plt.imshow(upsharpMask, cmap='gray')
    ax3.set_title("矩阵乘法")
    ax4 = plt.subplot(2, 2, 4)
    plt.imshow(imageSharp, cmap='gray')
    ax4.set_title("矩阵加法")
    plt.savefig('./result/' + 'hight_pass_filter' + pic, format='jpg')


def bandpass_filter(pic, D0=30, W=0, N=2, type='lhp', filter='ideal'):
    """
    带阻滤波器(bandstop filters，简称BSF)是指能通过大多数频率分量，但将某些范围的频率分量衰减到
    极低水平的滤波器，与带通滤波器的概念相比。其中点阻滤波器(notch filter)是一种特殊的带阻滤波
    器，它的阻带范围极小，有着很高的Q值(Q Factor)。

    Args:
        pic (str): 图像路径
        D0 (int, optional): 截止频率. Defaults to 30.
        W (int, optional): 带宽. Defaults to 0.
        N (int, optional): butterworth和指数滤波器的阶数. Defaults to 2.
        type (str, optional): lhp, bp, bs即低通和高通、带通、带阻. Defaults to 'lhp'.
        filter (str, optional): butterworth、ideal、exponential即巴特沃斯、理想、指数滤波器. Defaults to 'ideal'.

    Return:
        带通带阻滤波器处理后图像
    """
    image = cv.imread(pic, 0)
    dft = cv.dft(np.float32(image), flags=cv.DFT_COMPLEX_OUTPUT)  # 离散傅里叶变换
    dtf_shift = np.fft.fftshift(dft)
    rows, cols = image.shape[0], image.shape[1]
    crow, ccol = int(rows / 2), int(cols / 2)  # 计算频谱中心
    mask = np.ones((rows, cols, 2))  # 生成rows行cols列的2维矩阵
    maskk = np.ones((rows, cols, 2))  # 生成rows行cols列的2维矩阵

    for i in range(rows):
        for j in range(cols):
            D = np.sqrt((i - crow)**2 + (j - ccol)**2)
            if (filter.lower() == 'butterworth'):  # 巴特沃滤波器
                if (type == 'lhp'):
                    mask[i, j] = 1 / (1 + (D / D0)**(2 * N))
                    maskk[i, j] = 1 / (1 + (D0 / D)**(2 * N))
                elif (type == 'bsp'):
                    mask[i, j] = 1 / (1 + (D * W / (D**2 - D0**2))**(2 * N))
                    maskk[i, j] = 1 / (1 + ((D**2 - D0**2) / D * W)**(2 * N))
                else:
                    assert ('type error')

            elif (filter.lower() == 'ideal'):  # 理想滤波器
                if (type == 'lhp'):
                    if (D > D0):
                        mask[i, j] = 0
                    if (D < D0):
                        maskk[i, j]
                elif (type == 'bsp'):
                    if (D > D0 and D < D0 + W):
                        mask[i, j] = 0
                    if (D < D0 and D > D0 + W):
                        maskk[i, j] = 0
                else:
                    assert ('type error')

            elif (filter.lower() == 'exponential'):  # 指数滤波器
                if (type == 'lhp'):
                    mask[i, j] = np.exp(-(D / D0)**(2 * N))
                    maskk[i, j] = np.exp(-(D0 / D)**(2 * N))
                elif (type == 'bsp'):
                    mask[i, j] = np.exp(-(D * W / (D**2 - D0**2))**(2 * N))
                    maskk[i, j] = np.exp(-((D**2 - D0**2) / D * W)**(2 * N))
                else:
                    assert ('type error')

    lfshift = dtf_shift * mask
    hfshift = dtf_shift * maskk

    # 低通带阻滤波器
    lf_ishift = np.fft.ifftshift(lfshift)
    limg_back = cv.idft(lf_ishift)
    limg_back = cv.magnitude(limg_back[:, :, 0], limg_back[:, :, 1])
    limg_back = np.abs(limg_back)
    limg_back = (limg_back - np.amin(limg_back)) / (np.amax(limg_back) -
                                                    np.amin(limg_back))

    # 高通带阻滤波器
    hf_ishift = np.fft.ifftshift(hfshift)
    himg_back = cv.idft(hf_ishift)
    himg_back = cv.magnitude(himg_back[:, :, 0], himg_back[:, :, 1])
    himg_back = np.abs(himg_back)
    himg_back = (himg_back - np.amin(himg_back)) / (np.amax(himg_back) -
                                                    np.amin(himg_back))

    ax1 = plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    ax1.set_title("原图")
    ax2 = plt.subplot(1, 3, 2)
    plt.imshow(limg_back, cmap='gray')
    ax2.set_title("低通带阻滤波器")
    ax3 = plt.subplot(1, 3, 3)
    plt.imshow(himg_back, cmap='gray')
    ax3.set_title("高通带阻滤波器")
    plt.savefig('./result/' + 'bandpass_filter' + pic, format='jpg')


def homomorphic_filter(pic, d0=10, r1=0.5, rh=2, c=4, h=2.0, l=0.5):
    """
    同态滤波：对于一幅由物理过程产生的图像f(x,y)，可以表示为照射分量i(x,y)和反射分量r(x,y)的乘积。
    0<i(x,y)<∞，0<r(x,y)<1。i(x,y)描述景物的照明，变化缓慢，处于低频成分。r(x,y)描述景物的细节，变化较快，处于高频成分。

    因为该性质是乘性的，所以不能直接使用傅里叶变换对i(x,y)和r(x,y)进行控制，因此可以先对f(x,y)取对数，分离i(x,y)和r(x,y)。
    令z(x,y) = ln f(x,y) = ln i(x,y) + ln r(x,y)。在这个过程中，由于f(x,y)的取值范围为[0, L-1]，
    为了避免出现ln(0)的情况，故采用ln ( f(x,y) + 1 ) 来计算。
    
    然后取傅里叶变换，得到 Z(u,v) = Fi(u,v) + Fr(u,v)。
    然后使用一个滤波器，对Z(u,v)进行滤波，有 S(u,v) = H(u,v) Z(u,v) = H(u,v)Fi(u,v) + H(u,v)Fr(u,v)。
    滤波后，进行反傅里叶变换，有 s(x, y) = IDFT( S(u,v) )。
    最后，反对数（取指数），得到最后处理后的图像。g(x,y) = exp^(s(x,y)) = i0(x,y)+r0(x,y)。
    由于我们之前使用ln ( f(x,y)+1)，因此此处使用exp^(s(x,y)) - 1。 i0(x,y)和r0(x,y)分别是处理后图像的照射分量和入射分量。


    Args:
        pic ([type]): [description]
        d0 (int, optional): [description]. Defaults to 10.
        r1 (float, optional): [description]. Defaults to 0.5.
        rh (int, optional): [description]. Defaults to 2.
        c (int, optional): [description]. Defaults to 4.
        h (float, optional): [description]. Defaults to 2.0.
        l (float, optional): [description]. Defaults to 0.5.

    Return:
        同态滤波器处理后图像
    """
    src = cv.imread(pic, 0)
    gray = src.copy()
    if len(src.shape) > 2:
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    gray = np.float64(gray)
    rows, cols = gray.shape
    gray_fft = np.fft.fft2(gray)
    gray_fftshift = np.fft.fftshift(gray_fft)
    dst_fftshift = np.zeros_like(gray_fftshift)
    M, N = np.meshgrid(np.arange(-cols // 2, cols // 2),
                       np.arange(-rows // 2, rows // 2))
    D = np.sqrt(M**2 + N**2)
    Z = (rh - r1) * (1 - np.exp(-c * (D**2 / d0**2))) + r1
    dst_fftshift = Z * gray_fftshift
    dst_fftshift = (h - l) * dst_fftshift + l
    dst_ifftshift = np.fft.ifftshift(dst_fftshift)
    dst_ifft = np.fft.ifft2(dst_ifftshift)
    dst = np.real(dst_ifft)
    dst = np.uint8(np.clip(dst, 0, 255))
    ax1 = plt.subplot(1, 2, 1)
    plt.imshow(src, cmap='gray')
    ax1.set_title("原图")
    ax2 = plt.subplot(1, 2, 2)
    plt.imshow(dst, cmap='gray')
    ax2.set_title("同态滤波器")
    plt.savefig('./result/' + 'homomorphic_filter' + pic, format='jpg')


def blurr(pic):
    """
    使用空域模板进行的图像处理，被称为空域滤波。
    模板本身被称为空域滤波器。空域滤波的机理就是在待处理的图像中逐点地移动模板，
    滤波器在该点地响应通过事先定义的滤波器系数 与滤波模板扫过区域的相应像素值的关系来计算。

    Args:
        pic (str): 图像路径
    """
    image = cv.imread(pic)
    source = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    result = cv.blur(source, (5, 5))  # 均值滤波

    ax1 = plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    ax1.set_title("原图")
    ax2 = plt.subplot(1, 2, 2)
    plt.imshow(result, cmap='gray')
    ax2.set_title("空域噪声滤波器")
    plt.savefig('./result/' + 'blurr' + pic, format='jpg')


def ConbineFilter(pic):
    """
    组合滤波器

    Args:
        pic (str): 图像路径
    """
    image = cv.imread(pic, 0)
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    s1 = np.log(np.abs(fshift))
    w, h = image.shape
    flt = np.zeros(image.shape)

    rx1 = w / 4
    ry1 = h / 2
    rx2 = w * 3 / 4
    ry2 = h / 2

    r = min(w, h) / 6
    for i in range(1, w):
        for j in range(1, h):
            if ((i - rx1)**2 +
                (j - ry1)**2 >= r**2) and ((i - rx2)**2 +
                                           (j - ry2)**2 >= r**2):
                flt[i, j] = 1

    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * flt)))

    plt.subplots_adjust(left=0.0, bottom=0.0, top=1, right=2)
    ax1 = plt.subplot(1, 4, 1)
    plt.imshow(image, cmap='gray')
    ax1.set_title("原图")
    ax2 = plt.subplot(1, 4, 2)
    plt.imshow(s1, cmap='gray')
    ax2.set_title("中心频率域")
    ax3 = plt.subplot(1, 4, 3)
    plt.imshow(flt, cmap='gray')
    ax3.set_title("陷波滤波器")
    ax4 = plt.subplot(1, 4, 4)
    plt.imshow(new_img, cmap='gray')
    ax4.set_title("滤波后图像")
    plt.savefig('./result/' + 'ConbineFilter' + pic, format='jpg')


def NoBineFilter(pic):
    """
    无约束滤波器
    逆滤波复原过程：对退化的图像进行二位傅里叶变换，计算系统点扩散函数的二位傅里叶变换，
    引入H（fx,fy）计算并且对结果进行逆傅里叶变换。

    Args:
        pic (str): 图像路径
    """
    image = cv.imread(pic)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    dft = cv.dft(np.float32(image), flags=cv.DFT_COMPLEX_OUTPUT)  # 傅里叶变换
    dftshift = np.fft.fftshift(dft)
    res1 = 20 * np.log(cv.magnitude(dftshift[:, :, 0], dftshift[:, :, 1]))

    ishift = np.fft.ifftshift(dftshift)  # 傅里叶逆变换
    iimg = cv.idft(ishift)
    res2 = cv.magnitude(iimg[:, :, 0], iimg[:, :, 1])

    ax1 = plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    ax1.set_title("原图")
    ax2 = plt.subplot(1, 3, 2)
    plt.imshow(res1, cmap='gray')
    ax2.set_title("傅里叶变换")
    ax3 = plt.subplot(1, 3, 3)
    plt.imshow(res2, cmap='gray')
    ax3.set_title("傅里叶逆变换")
    plt.savefig('./result/' + 'NoBineFilter' + pic, format='jpg')


def Weipingmian(pic):
    """
    位平面编码
    比特平面编码又被称为位平面编码，位平面编码是一种通过单独地处理图像的位平面来减少像素间冗余
    地有效技术，它将一副多级图像分解为一系列二值图像并采用几种熟知地二值图像压缩方法来对每一幅
    二值图像进行压缩。位平面编码分为两个步骤：位平面分解和位平面编码。

    Args:
        pic (str): 图像路径
    """
    image = cv.imread(pic, 0)
    rows, cols = image.shape[0], image.shape[1]
    x = np.zeros((rows, cols, 8), dtype=np.uint8)  # 构造提取矩阵
    for i in range(8):
        x[:, :, i] = 2**i
    w = np.zeros((rows, cols, 8), dtype=np.uint8)

    for i in range(8):
        w[:, :, i] = cv.bitwise_and(image, x[:, :, i])  # 提取位平面
        mask = w[:, :, i] > 0  # 阈值处理
        w[mask] = 255
        plt.subplots_adjust(left=0.0, bottom=0.0, top=1, right=2)
        plt.subplot(1, 8, i + 1), plt.xticks([]), plt.yticks([]), plt.title(i)
        plt.imshow(w[:, :, i], cmap='gray')

    plt.savefig('./result/' + 'Weipingmian' + pic, format='jpg')


def DPCM_funticon(pic):
    """
    DPCM编码
    DPCM编码，简称差值编码，是对模拟信号幅度抽样的差值进行量化编码的调制方式。这种方式是用已
    经过去的抽样值来预测当前的抽样值，对它们的差值进行编码。差值编码可以提高编码效率，这种技术
    已应用于模拟信号的数字通信中。对于有些信号(例如图像信号)由于信号的瞬间斜率比较大，容易引起过
    载，因此，不能用简单增量调制进行编码，除此之外，这类信号也没有像话音信号那种音节特性，因而
    也不能采用像音节压扩的方法，只能采用瞬时压扩的方法。但瞬时压扩实现起来比较困难，因此，对于
    这类瞬时斜率比较大的信号，通常采用一种综合了增量调制和脉冲编码调制两者特点的调制方法进行编
    码，这种编码方式被简称为脉码增量调制，或称差值脉码调制，用DPCM表示。

    Args:
        pic (str): 图像路径
    """
    image = cv.imread(pic, 1)
    grayimg = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    rows = grayimg.shape[0]
    cols = grayimg.shape[1]

    image1 = grayimg.flatten()  #把灰度化后的二维图像降维成一维列表

    for i in range(len(image1)):
        if image1[i] >= 200:
            image1[i] = 255
        if image1[i] < 200:
            image1[i] = 0
    data = []
    image3 = []
    count = 1

    for i in range(len(image1) - 1):
        if (count == 1):
            image3.append(image1[i])
        if image1[i] == image1[i + 1]:
            count = count + 1
            if i == len(image1) - 2:
                image3.append(image1[i])
                data.append(count)
        else:
            data.append(count)
            count = 1
    if (image1[len(image1) - 1] != image1[-1]):
        image3.append(image1[len(image1) - 1])
        data.append(1)

    rec_image = []
    for i in range(len(data)):
        for j in range(data[i]):
            rec_image.append(image3[i])
    rec_image = np.reshape(rec_image, (rows, cols))  # 行程编码解码

    ax1 = plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    ax1.set_title("原图")
    ax2 = plt.subplot(1, 2, 2)
    plt.imshow(rec_image, cmap='gray')
    ax2.set_title("DPCM编码")
    plt.savefig('./result/' + 'DPCM_funticon' + pic, format='jpg')


def Dct(pic):
    """
    余弦变换编码
    DCT变换的基本思路是将图像分解为8×8的子块或16×16的子块，并对每一个子块进行单独的DCT变换，
    然后对变换结果进行量化、编码。随着子块尺寸的增加，算法的复杂度急剧上升，
    因此，实用中通常采用8×8的子块进行变换，但采用较大的子块可以明显减少图像分块效应。

    Args:
        pic (str): 图像路径
    """
    image = cv.imread(pic)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    img_dct = cv.dct(np.array(image, np.float32))
    img_dct[0:100, 0:100] = 0
    img_idct = np.array(cv.idct(img_dct), np.uint8)
    dct_out = img_idct

    ax1 = plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    ax1.set_title("原图")
    ax2 = plt.subplot(1, 2, 2)
    plt.imshow(dct_out, cmap='gray')
    ax2.set_title("余弦变换编码")
    plt.savefig('./result/' + 'Dct' + pic, format='jpg')


def tool_Denoising(inputGrayPic, value):
    """
    去噪

    Args:
        inputGrayPic (str): 灰度图像路径
        value (int): 去噪阈值

    """
    result = inputGrayPic
    height = result.shape[0]
    weight = result.shape[1]
    for row in range(height):
        for col in range(weight):
            if (abs(result[row, col]) > value):
                result[row, col] = 0  #频率的数值0为低频
    return result


def WaTf(pic):
    """
    小波变换编码
    小波变换（Wavelet Transfom）编码是数字地球的最有发展前途的数据压缩方法。小波分析优于傅立叶
    分析的方面是∶ 它的时间域或频率域同时具有良好的局部化性质，而且由于对高频成分采用逐渐精细的
    时域或空域取样步长，从而可以聚集到对象任意细节，所以它称为"数字显微镜"。小波分析的优势在于
    可以同时进行时频域分析，比傅里叶变换更适合处理非平稳信号。

    Args:
        pic (str): 图像路径
    """
    image = cv.imread(pic, 0)
    # cA，cH,cV,cD 分别为近似分量(低频分量)、水平细节分量、垂直细节分量和对角细节分量
    cA, (cH, cV, cD) = dwt2(image, 'haar')  # dwt2函数第二个参数指定小波基
    VALUE = 60  # 设置去噪阈值
    cH = tool_Denoising(cH, VALUE)  # 处理水平高频
    cV = tool_Denoising(cV, VALUE)  # 处理垂直高频
    cD = tool_Denoising(cD, VALUE)  # 处理对角线高频
    rebuild = idwt2((cA, (cH, cV, cD)), 'haar')  # 重构图像

    plt.subplots_adjust(left=0.0, bottom=0.0, top=1, right=2)
    ax1 = plt.subplot(1, 6, 1)
    plt.imshow(image)
    ax1.set_title("原图")
    ax2 = plt.subplot(1, 6, 2)
    plt.imshow(cA)
    ax2.set_title("CA")
    ax3 = plt.subplot(1, 6, 3)
    plt.imshow(cH)
    ax3.set_title("CH")
    ax4 = plt.subplot(1, 6, 4)
    plt.imshow(cV)
    ax4.set_title("CV")
    ax5 = plt.subplot(1, 6, 5)
    plt.imshow(cD)
    ax5.set_title("CD")
    ax6 = plt.subplot(1, 6, 6)
    plt.imshow(rebuild)
    ax6.set_title("rebuild")
    plt.savefig('./result/' + 'WaTf' + pic, format='jpg')


def image_dynamic(pic):
    """
    动态规划
    动态规划经常用于求解某些具有最优性质的问题，如今随着对动态规划算法的日渐深入的研究.动态规划
    被用在生产调度等各个方面。

    Args:
        pic (str): 图像路径
    """
    image = cv.imread(pic)
    fil = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])
    res = cv.filter2D(image, -1, fil)

    ax1 = plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    ax1.set_title("原图")
    ax2 = plt.subplot(1, 2, 2)
    plt.imshow(res, cmap='gray')
    ax2.set_title("动态规划")
    plt.savefig('./result/' + 'image_dynamic' + pic, format='jpg')


def Threshold_seg(pic):
    """
    单阈值分割
    阈值分割最简单的方法就是人工选择法。基于灰度阈值的分割方法，其关键是如何
    人工选择方法是通过人眼的观察，应用人对图像的知识，在分析图像直方图的基础上，
    人工选择出合理的阈值。也可以在人工选择出阈值后，根据分割的效果，
    不断地进行交互操作，从而选择出最佳的阈值。

    Args:
        pic (str): 图像路径
    """
    image = cv.imread(pic)
    rows = image.shape[0]
    cols = image.shape[1]

    grayHist = np.zeros([256], np.uint64)
    for r in range(rows):
        for c in range(cols):
            grayHist[image[r][c]] += 1
            histogram = grayHist
    maxLoc = np.where(histogram == np.max(histogram))
    firstPeak = maxLoc[0][0]  #
    measureDists = np.zeros([256], np.float32)

    for k in range(256):
        kkk = np.array(k - firstPeak)
        measureDists[k] = pow(kkk, 2) * histogram[k]

    maxLoc2 = np.where(measureDists == np.max(measureDists))
    secondPeak = maxLoc2[0][0]

    if firstPeak > secondPeak:
        temp = histogram[int(secondPeak):int(firstPeak)]
        minLoc = np.where(temp == np.min(temp))
        thresh = secondPeak + minLoc[0][0] + 1
    else:
        temp = histogram[int(firstPeak):int(secondPeak)]
        minLoc = np.where(temp == np.min(temp))
        thresh = firstPeak + minLoc[0][0] + 1
    threshImage_out = image.copy()
    threshImage_out[threshImage_out > thresh] = 255
    threshImage_out[threshImage_out <= thresh] = 0

    ax1 = plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    ax1.set_title("原图")
    ax2 = plt.subplot(1, 2, 2)
    plt.imshow(threshImage_out, cmap='gray')
    ax2.set_title("单阈值分割")
    plt.savefig('./result/' + 'Threshold_seg' + pic, format='jpg')


def img_extraction(image):
    """
    利用susan角点检测算法，对图像进行处理

    Args:
        image (str): 图像路径
    """
    threshold_value = (int(image.max()) - int(image.min())) / 10  # 阈值
    offsetX = [
        -1, 0, 1, -2, -1, 0, 1, 2, -3, -2, -1, 0, 1, 2, 3, -3, -2, -1, 0, 1, 2,
        3, -3, -2, -1, 0, 1, 2, 3, -2, -1, 0, 1, 2, -1, 0, 1
    ]
    offsetY = [
        -3, -3, -3, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3
    ]

    # 利用圆形模板遍历图像，计算每点处的USAN值
    for i in range(3, image.shape[0] - 3):
        for j in range(3, image.shape[1] - 3):
            same = 0
            for k in range(0, 37):
                if abs(
                        int(image[i + int(offsetY[k]), j + int(offsetX[k]),
                                  0]) - int(image[i, j, 0])) < threshold_value:
                    same += 1

            if same < 18:
                image[i, j, 0] = 18 - same
                image[i, j, 1] = 18 - same
                image[i, j, 2] = 18 - same
            else:
                image[i, j, 0] = 0
                image[i, j, 1] = 0
                image[i, j, 2] = 0


def img_revise(image):
    """
    用于对角处理后的图像，进行非极大值抑制修正

    Args:
        image (str): 图像路径
    """
    X = [-1, -1, -1, 0, 0, 1, 1, 1]  # X轴偏移
    Y = [-1, 0, 1, -1, 1, -1, 0, 1]  # Y轴偏移

    for i in range(4, image.shape[0] - 4):
        for j in range(4, image.shape[1] - 4):
            flag = 0
            for k in range(0, 8):
                if image[i, j, 0] <= image[int(i + X[k]), int(j + Y[k]), 0]:
                    flag += 1
                    break
                if flag == 0:  # 判断是否是周围8个点中最大的那个值，是的话则保留
                    image[i, j, 0] = 255
                    image[i, j, 1] = 255
                    image[i, j, 2] = 255
                else:
                    image[i, j, 0] = 0
                    image[i, j, 1] = 0
                    image[i, j, 2] = 0


def susan(pic):
    image = cv.imread(pic)
    pre = image.copy()
    img_extraction(pre)
    img_revise(pre)
    ax1 = plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    ax1.set_title("原图")
    ax2 = plt.subplot(1, 2, 2)
    plt.imshow(pre, cmap='gray')
    ax2.set_title("单阈值分割")
    plt.savefig('./result/' + 'susan' + pic, format='jpg')


def CV(LSF, img, nu, mu, epison, step):
    Drc = (epison / math.pi) / (epison * epison + LSF * LSF)
    #Hea = 0.5*(1+(2/math.pi)*mat_math(LSF/epison,'atan'))
    Hea = 0.5 * (1 + (2 / math.pi) * np.arctan(LSF / epison))
    Iy, Ix = np.gradient(LSF)  ##q4#
    #s = mat_math(Ix*Ix+Iy*Iy,"sqrt")
    s = np.sqrt(Ix * Ix + Iy * Iy)
    Nx = Ix / (s + 0.000001)
    Ny = Iy / (s + 0.000001)
    Mxx, Nxx = np.gradient(Nx)
    Nyy, Myy = np.gradient(Ny)
    cur = Nxx + Nyy

    Length = nu * Drc * cur
    Area = mu * Drc

    s1 = Hea * img
    s2 = (1 - Hea) * img
    s3 = 1 - Hea
    C1 = s1.sum() / Hea.sum()
    C2 = s2.sum() / s3.sum()

    CVterm = Drc * (-1 * (img - C1) * (img - C1) + 1 * (img - C2) * (img - C2))
    LSF = LSF + step * (Length + Area + CVterm)
    return LSF


def FiAction(pic):
    """
    主动轮廓模型
    将图像分割问题转换为求解能量泛函最小值问题，为图像分割提供一种全新的思路，称为研究的重点和热点。
    主动轮廓模型的主要原理通过构造能量泛函，在能量函数最小值驱动下，轮廓曲线逐渐向待检测物体的边缘逼近，最终分割出目标。
    由于主动轮廓模型利用曲线演化定位目标的边缘，因此也称为Snake模型。主动轮廓模型是当前应用最多的利用变分思想求解的图像分割方法。
    其最大优点是在高噪声的情况下，也能得到连续、光滑的闭合分割边界。
    按照能量函数构造方式的不同，可以将主动轮廓模型主要分为基于边缘和基于区域两类，
    同时也有一些研究人员提出了基于边缘和区域相结合的主动轮廓模型。

    Args:
        pic (str): 图像路径
    """
    Image = cv.imread(pic, 1)
    image = cv.cvtColor(Image, cv.COLOR_BGR2GRAY)
    img = np.array(image, dtype='float64')

    IniLSF = np.ones([img.shape[0], img.shape[1]], img.dtype)  # 初始化水平集函数
    IniLSF[40:80, 40:80] = -1
    IniLSF = IniLSF

    nu = 0.0001 * 255 * 255
    mu = 1
    num = 10
    epison = 1
    step = 0.1
    LSF = -IniLSF
    ax1 = plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    ax1.set_title("原图")
    for i in range(1, num):
        LSF = CV(LSF, img, nu, mu, epison, step)
        if i % 2 == 0:
            ax2 = plt.subplot(1, 2, 2)
            plt.contour(LSF, [0],
                        linewidths=3.0,
                        linestyles='dotted',
                        colors='r')
    ax2.set_title("主动轮廓")
    plt.savefig('./result/' + 'FiAction' + pic, format='jpg')


def WaterShed(pic):
    """
    分水岭分割
    分水岭算法的整个过程:
    1. 把梯度图像中的所有像素按照灰度值进行分类，并设定一个测地距离阈值。
    2. 找到灰度值最小的像素点（默认标记为灰度值最低点），让threshold从最小值开始增长，这些点为起始点。
    3. 水平面在增长的过程中，会碰到周围的邻域像素，测量这些像素到起始点（灰度值最低点）的测地距离，
        如果小于设定阈值，则将这些像素淹没，否则在这些像素上设置大坝，这样就对这些邻域像素进行了分类。
    4. 随着水平面越来越高，会设置更多更高的大坝，直到灰度值的最大值，所有区域都在分水岭线上相遇，
        这些大坝就对整个图像像素的进行了分区。

    Args:
        pic (str): 图像路径
    """
    image = cv.imread(pic)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(gray, 0, 255,
                               cv.THRESH_BINARY_INV + cv.THRESH_OTSU)  # 阈值分割
    kernel = np.ones((3, 3), np.uint8)  # 开运算
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv.dilate(opening, kernel, iterations=3)
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)  # 获取前景区域
    ret, sure_fg = cv.threshold(dist_transform, 0.1 * dist_transform.max(),
                                255, 0)
    sure_fg = np.uint8(sure_fg)
    unknow = cv.subtract(sure_bg, sure_fg)
    ret, markers = cv.connectedComponents(sure_fg, connectivity=8)  # 连通区域处理
    markers = markers + 1
    markers[unknow == 255] = 0  # 去掉属于背景区域的部分
    watershed = image.copy()
    markers = cv.watershed(watershed, markers)  # 分水岭算法
    watershed[markers == -1] = [255, 0, 0]

    ax1 = plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    ax1.set_title("原图")
    ax2 = plt.subplot(1, 3, 2)
    plt.imshow(markers, cmap='gray')
    ax2.set_title("去掉背景区域")
    ax3 = plt.subplot(1, 3, 3)
    plt.imshow(watershed, cmap='gray')
    ax3.set_title("分水岭分割")
    plt.savefig('./result/' + 'WaterShed' + pic, format='jpg')


def collect(pic):
    """
    腐蚀：是求局部最小值，移动核B，内核在图像中滑动（如二维卷积）。只有当内核下的所有像素都为 1
        时，原始图像中的像素（1 或 0）才会被视为 1，否则会被侵蚀（变为零）。所以根据内核的大小，边界
        附近的所有像素都将被丢弃。
    膨胀：就是求局部最大值的操作，核B与图形卷积，即计算核B覆盖的区域的像素点的最大值，并把这个最
        大值赋值给参考点指定的像素。如果内核下至少有一个像素为“1”，则像素元素为“1”。所以它会增加图像
        中的白色区域，或者增加前景对象的大小。
    开运算：开操作是先腐蚀后膨胀，它能够消除噪音，在纤细处分离物体和平滑较大物体边界的作用。
    闭运算：闭操作是先膨胀后腐蚀，它在填充前景对象内的小孔或者对象上的小黑点时很有用。

    Args:
        pic (str): 图像路径
    """
    image = cv.imread(pic)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    blur = cv.blur(image, (5, 5))  # 腐蚀
    dilated = cv.dilate(image, kernel)  # 膨胀
    opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)  # 开操作
    closing = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)  # 闭操作

    plt.subplots_adjust(left=0.0, bottom=0.0, top=1, right=2)
    ax1 = plt.subplot(1, 5, 1)
    plt.imshow(image)
    ax1.set_title("原图")
    ax2 = plt.subplot(1, 5, 2)
    plt.imshow(blur)
    ax2.set_title("腐蚀")
    ax3 = plt.subplot(1, 5, 3)
    plt.imshow(dilated)
    ax3.set_title("膨胀")
    ax4 = plt.subplot(1, 5, 4)
    plt.imshow(opening)
    ax4.set_title("开操作")
    ax5 = plt.subplot(1, 5, 5)
    plt.imshow(closing)
    ax5.set_title("闭操作")
    plt.savefig('./result/' + 'collect' + pic, format='jpg')


def ConbineS(pic):
    """
    基于二值形态学的应用
    噪声去噪：使用开操作的方式，先腐蚀后膨胀。可以消除小的噪点，填充孤立的洞。能排除小区域物体、消除孤立
        点、去噪和平滑物体的轮廓。
    目标检测：可以使用梯度的方式，提取出图像的轮廓。梯度的原理是对图像分别使用膨胀和腐蚀，然后计算两者之
        间的差值，即可提取出图像的边界。
    区域填充：我们只需要用膨胀后的图像与边界的补图像进行相交，就能把膨胀限制在边界内部，直到我们的膨胀图像
        B填充满边界A，这时候取AB并集，就是最终的区域填充结果。

    Args:
        pic (str): 图像路径
    """
    image = cv.imread(pic)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (40, 40))
    gradient = cv.morphologyEx(image, cv.MORPH_GRADIENT, kernel)
    noise = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
    copyImg = image.copy()
    rows = image.shape[0] + 2
    cols = image.shape[1] + 2
    rrr = rows // 2
    ccc = cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    cv.floodFill(copyImg, mask, (rrr, ccc), (0, 255, 255), (100, 100, 100),
                 (50, 50, 50), cv.FLOODFILL_FIXED_RANGE)

    plt.subplots_adjust(left=0.0, bottom=0.0, top=1, right=2)
    ax1 = plt.subplot(1, 4, 1)
    plt.imshow(image)
    ax1.set_title("原图")
    ax2 = plt.subplot(1, 4, 2)
    plt.imshow(noise)
    ax2.set_title("噪声去噪")
    ax3 = plt.subplot(1, 4, 3)
    plt.imshow(gradient)
    ax3.set_title("目标检测")
    ax4 = plt.subplot(1, 4, 4)
    plt.imshow(copyImg)
    ax4.set_title("区域填充")
    plt.savefig('./result/' + 'ConbineS' + pic, format='jpg')


def conbineGray(pic):
    """
    基于灰度形态学的应用
    形态梯度：梯度用于刻画目标边界或边缘位于图像灰度级剧烈变化的区域、形态学梯度根据
        膨胀或者腐蚀与原图作差组合来实现增强结构元素领域中像素的强度，突出高亮区域的外围。
    形态平滑：图像平滑又被称为图像模糊，用于消除图片中的噪声。
    高帽：高帽运算是原图像和原图像开运算结果的差值。
    黑帽：黑帽运算是原图像和原图像闭运算的差值。


    Args:
        pic (str): 图像路径
    """
    image = cv.imread(pic)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (20, 30))

    Gd_out = cv.morphologyEx(gray, cv.MORPH_GRADIENT, kernel)
    sm = cv.boxFilter(gray, -1, (3, 3), normalize=True)

    hat_g_out = cv.morphologyEx(gray, cv.MORPH_TOPHAT, kernel)

    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    hat_b_out = cv.morphologyEx(binary, cv.MORPH_BLACKHAT, kernel)

    plt.subplot(221), plt.axis("off"), plt.title('Gradient'), plt.imshow(
        Gd_out, cmap='gray')
    plt.subplot(222), plt.axis("off"), plt.title("Smooth"), plt.imshow(
        sm, cmap='gray')
    plt.subplot(223), plt.axis("off"), plt.title("HightHat"), plt.imshow(
        hat_g_out, cmap='gray')
    plt.subplot(224), plt.axis("off"), plt.title("BlackHat"), plt.imshow(
        hat_b_out, cmap='gray')
    plt.savefig('./result/' + '9d' + pic, format='jpg')

    plt.subplots_adjust(left=0.0, bottom=0.0, top=1, right=2)
    ax1 = plt.subplot(1, 5, 1)
    plt.imshow(image)
    ax1.set_title("原图")
    ax2 = plt.subplot(1, 5, 2)
    plt.imshow(Gd_out)
    ax2.set_title("形态梯度")
    ax3 = plt.subplot(1, 5, 3)
    plt.imshow(sm)
    ax3.set_title("形态平滑")
    ax4 = plt.subplot(1, 5, 4)
    plt.imshow(hat_g_out)
    ax4.set_title("高帽变换")
    ax5 = plt.subplot(1, 5, 5)
    plt.imshow(hat_b_out)
    ax5.set_title("黑帽变换")
    plt.savefig('./result/' + 'conbineGray' + pic, format='jpg')


def Sobel_filter(pic):
    """
    Sobel算子主要用于边缘检测，在技术上它是以离散型的差分算子，用来运算图像亮度函数的梯度的近似值。
    Sobel算子是典型的基于一阶导数的边缘检测算子，由于该算子中引入了类似局部平均的运算，因此对噪声具有平滑作用，能很好的消除噪声的影响。
    Sobel算子对于象素的位置的影响做了加权，与Prewitt算子、Roberts算子相比因此效果更好。
    Sobel算子包含两组3x3的矩阵，分别为横向及纵向模板，将之与图像作平面卷积，即可分别得出横向及纵向的亮度差分近似值。

    Args:
        pic (str): 图像路径
    """
    image = cv.imread(pic)
    x = cv.Sobel(image, cv.CV_16S, 1, 0)
    y = cv.Sobel(image, cv.CV_16S, 0, 1)
    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)
    sobel_out = cv.addWeighted(absX, 0.5, absY, 0.5, 0)

    ax1 = plt.subplot(1, 2, 1)
    plt.imshow(image)
    ax1.set_title("原图")
    ax2 = plt.subplot(1, 2, 2)
    plt.imshow(sobel_out)
    ax2.set_title("Sobel")
    plt.savefig('./result/' + 'Sobel_filter' + pic, format='jpg')


def Roberts_filter(pic):
    """
    Roberts算子又称为交叉微分算法，它是基于交叉差分的梯度算法，通过局部差分计算检测边缘线条。
    常用来处理具有陡峭的低噪声图像，当图像边缘接近于正45度或负45度时，该算法处理效果更理想。
    其缺点是对边缘的定位不太准确，提取的边缘线条较粗。

    Args:
        pic (str): 图像路径
    """
    image = cv.imread(pic)
    grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Roberts
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)
    x = cv.filter2D(grayImage, cv.CV_16S, kernelx)
    y = cv.filter2D(grayImage, cv.CV_16S, kernely)
    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)
    roberts_out = cv.addWeighted(absX, 0.5, absY, 0.5, 0)

    ax1 = plt.subplot(1, 2, 1)
    plt.imshow(image)
    ax1.set_title("原图")
    ax2 = plt.subplot(1, 2, 2)
    plt.imshow(roberts_out)
    ax2.set_title("Roberts")
    plt.savefig('./result/' + 'Roberts_filter' + pic, format='jpg')


def Laplacian_filter(pic):
    """
    Laplace算子作为边缘检测之一，和Sobel算子一样也是工程数学中常用的一种积分变换，属于空间锐化滤波操作。
    拉普拉斯算子（Laplace Operator）是n维欧几里德空间中的一个二阶微分算子，定义为梯度（▽f）的散度（▽·f）。
    拉普拉斯算子也可以推广为定义在黎曼流形上的椭圆型算子，称为拉普拉斯-贝尔特拉米算子。

    Args:
        pic (str): 图像路径
    """
    image = cv.imread(pic)
    grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    dst = cv.Laplacian(grayImage, cv.CV_16S, ksize=3)
    lap_out = cv.convertScaleAbs(dst)

    ax1 = plt.subplot(1, 2, 1)
    plt.imshow(image)
    ax1.set_title("原图")
    ax2 = plt.subplot(1, 2, 2)
    plt.imshow(lap_out)
    ax2.set_title("Laplacian")
    plt.savefig('./result/' + 'Laplacian_filter' + pic, format='jpg')


def Canny_filter(pic):
    """
    Canny 边缘检测的步骤：
        1. 消除噪声。
        2. 计算梯度幅值和方向。
        3. 非极大值抑制。
        4. 滞后阈值。

    Args:
        pic (str): 图像路径
    """
    image = cv.imread(pic)
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1)
    canny_out = cv.Canny(xgrad, ygrad, 50, 150)

    ax1 = plt.subplot(1, 2, 1)
    plt.imshow(image)
    ax1.set_title("原图")
    ax2 = plt.subplot(1, 2, 2)
    plt.imshow(canny_out)
    ax2.set_title("Canny")
    plt.savefig('./result/' + 'Canny_filter' + pic, format='jpg')


def Prewitt_filter(pic):
    """
    Prewitt算子是一种一阶微分算子边缘检测，利用像素点上下、左右邻点的灰度差，在边缘处达到极值检
    测边缘，去掉部分伪边缘，对噪声具有平滑作用。其原理是在图像空间利用两个方向模板与图像进行邻
    域卷积卷积来完成的，这两个方向模板一个检测水平边缘，一个检测垂直边缘。

    Args:
        pic (str): 图像路径
    """
    image = cv.imread(pic)
    grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Prewitt
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    x = cv.filter2D(grayImage, cv.CV_16S, kernelx)
    y = cv.filter2D(grayImage, cv.CV_16S, kernely)
    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)
    prewitt_out = cv.addWeighted(absX, 0.5, absY, 0.5, 0)

    ax1 = plt.subplot(1, 2, 1)
    plt.imshow(image)
    ax1.set_title("原图")
    ax2 = plt.subplot(1, 2, 2)
    plt.imshow(prewitt_out)
    ax2.set_title("Prewitt")
    plt.savefig('./result/' + 'Prewitt_filter' + pic, format='jpg')


def LoGaussian_filter(pic):
    """
    Laplacian of Gaussian
    拉普拉斯算子是图像二阶空间导数的二维各向同性测度。拉普拉斯算子可以突出图像中强度发生快速变化的区域，
    因此常用在边缘检测任务当中。在进行Laplacian操作之前通常需要先用高斯平滑滤波器对图像进行平滑处理，
    以降低Laplacian操作对于噪声的敏感性。该操作通常是输入一张灰度图，经过处理之后输出一张灰度图。

    Args:
        pic (str): 图像路径
    """
    image = cv.imread(pic)
    grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gaussian = cv.GaussianBlur(grayImage, (3,3), 0)
    dst = cv.Laplacian(gaussian, cv.CV_16S, ksize=3)
    loG_out = cv.convertScaleAbs(dst)

    ax1 = plt.subplot(1, 2, 1)
    plt.imshow(image)
    ax1.set_title("原图")
    ax2 = plt.subplot(1, 2, 2)
    plt.imshow(loG_out)
    ax2.set_title("Laplacian of Gaussian")
    plt.savefig('./result/' + 'LoGaussian_filter' + pic, format='jpg')