# :dizzy:基于Python OpenCV的图像算法研究

本项目旨在帮助您深入了解图像处理的各种算法，并使用 Python OpenCV 库实现这些算法。您可以在 main.py 文件中找到 60 多种图像处理算法的详细说明，并且各个函数功能高度解耦，可以单独运行。

## :rocket:研究背景

图像处理是计算机视觉和图像识别领域中非常重要的研究方向。随着深度学习算法的发展，图像处理技术在很多领域得到了广泛的应用，Python OpenCV 是一个开源的图像处理库，提供了大量的图像处理算法，是图像处理领域中的重要工具。

本项目旨在利用 Python OpenCV 库，研究多种图像处理算法的原理和实现方法，并实现了一个图像处理工具集。通过对图像处理算法的研究，不仅提高了我们的技术水平，也为以后的图像处理项目提供了重要的基础。

## :wrench:实验环境

- Python 3.7
- OpenCV 4.4.0
- Numpy

## 🤖运行示例

⚡运行本项目代码前请执行以下命令安装所依赖的库

```shell
pip install opencv-python
pip install numpy
pip install Pillow
```

Python 代码如下（直接运行）：

```python
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import os
from PIL import Image, ImageTk
import os

# 检查文件夹是否存在，如果不存在则创建一个文件夹
if not os.path.exists('result'):
    os.makedirs('result')

def select_image():
    """
    点击选择图片按钮时调用的函数，用于选择图像并显示在窗口中
    """
    global image_path
    # 使用filedialog模块弹出选择文件对话框，获取选择的图像路径
    image_path = filedialog.askopenfilename()
    # 使用OpenCV读取图像
    img = cv2.imread(image_path)
    # 将OpenCV读取的BGR图像转换为RGB图像
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 将图像转换为PIL图像
    img = Image.fromarray(img)
    # 将PIL图像转换为可以在Tkinter窗口中显示的图像
    img = ImageTk.PhotoImage(img)
    # 使用config()函数更新显示图像的标签的图像
    input_label.config(image=img)
    # 将图像存储在input_label变量中，以防图像被GC回收
    input_label.image = img

def process_image():
    # 声明全局变量 output_img
    global output_img
    # 使用 cv2.imread 读取图片
    img = cv2.imread(image_path)
    # 调用 translate_image 函数进行平移变换
    output_img = translate_image(img)
    # 将 BGR 格式图片转换为 RGB 格式
    output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    # 将 numpy 数组格式图片转换为 Image 对象
    output_img = Image.fromarray(output_img)
    # 将 Image 对象转换为 Tkinter 可用的 PhotoImage 对象
    output_img = ImageTk.PhotoImage(output_img)
    # 配置输出图片的 Label 组件，将处理后的图片显示在该组件上
    output_label.config(image=output_img)
    # 更新 output_label.image 属性的值
    output_label.image = output_img
    # 使用 cv2.imwrite 函数将处理后的图片保存到 result 文件夹中
    cv2.imwrite("result/result.jpg", output_img)


def translate_image(img):
    # 获取图像的行数和列数
    rows, cols = img.shape[:2]
    # 创建一个移动矩阵
    M = np.float32([[1,0,100],[0,1,50]])
    # 应用仿射变换到图像上
    output_img = cv2.warpAffine(img, M, (cols, rows))
    # 返回处理后的图像
    return output_img


# 创建窗口
root = tk.Tk()
# 设置窗口标题
root.title("图像处理器")

# 创建输入图像标签
input_label = tk.Label(root)
# 将标签放置在左侧
input_label.pack(side="left")

# 创建输出图像标签
output_label = tk.Label(root)
# 将标签放置在右侧
output_label.pack(side="right")

# 创建选择图像按钮
select_button = tk.Button(root, text="选择图像", command=select_image)
# 将按钮放置在窗口中
select_button.pack()

# 创建处理图像按钮
process_button = tk.Button(root, text="开始处理", command=process_image)
# 将按钮放置在窗口中
process_button.pack()

# 运行主循环
root.mainloop()
```

⚡ ***获取[完整版](https://mbd.pub/o/bread/YZqcm51r)查看所有60+种图像处理算法，包括但不限于：***

- 图像的坐标变换
    - [x] 平移变换
    - [x] 尺度变换
    - [x] 旋转变换
    - [x] 仿射变换
- 灰度映射
  - [x] 求反
  - [x] 动态范围压缩
  - [x] 阶梯量化
  - [x] 阈值分割
- 图像的算术运算
  - [x] 加法
  - [x] 平均法消除噪声
  - [x] 减法
- 直方图修正：
  - [x] 直方图均衡化
  - [x] 直方图规定化
- 空域滤波
  - [x] 线性平滑滤波器
  - [x] 线性锐化滤波器
  - [x] 非线性平滑滤波器
  - [x] 非线性锐化滤波器
- 傅里叶变换和反变换
  - [x] 傅里叶变换
  - [x] 反傅里叶变换
- 高通和低通滤波器
  - [x] 理想滤波器
  - [x] 巴特沃斯滤波器
  - [x] 指数滤波器
- 特殊高通滤波器
  - [x] 高频增强滤波器
  - [x] 高频提升滤波器
- [x] 带通带阻滤波器
- [x] 同态滤波器
- [x] 空域噪声滤波器
- [x] 组合滤波器
- [x] 无约束滤波器
- 变长编码
    - [x] 哈夫曼编码
    - [x] 哥伦布编码
    - [x] 香农-法诺编码
    - [x] 算数编码
- [x] 位平面编码
- 预测编码
    - [x] DPCM编码
    - [x] 余弦变换编码
    - [x] 小波变换编码
- 图像分割
    - [x] 动态规划
    - [x] 单阈值分割
- 典型分割
    - [x] SUSAN边缘检测
    - [x] 主动轮廓
    - [x] 分水岭分割
- 二值形态学
  - [x] 腐蚀
  - [x] 膨胀
  - [x] 开操作
  - [x] 闭操作
- 二值形态学的应用
  - [x] 噪声去燥
  - [x] 目标检测
  - [x] 区域填充
- 灰度形态学的应用
  - [x] 形态梯度
  - [x] 形态平滑
  - [x] 高帽
  - [x] 黑帽
- 算子
  - [x] Sobel算子
  - [x] Roberts算子
  - [x] Laplace算子
  - [x] Canny算子
  - [x] Prewitt算子
  - [x] 高斯拉普拉斯算子
  
## 😘基础版

[下载基础版](https://github.com/kangvcar/kkimage/releases/download/v1.2.0/kkapp-base.exe)

- 每次支持处理一个图像
- 支持的4种算法包括：平移变换、尺度变换、旋转变换、仿射变换

## 🥰完整版

[获取完整版](https://mbd.pub/o/bread/YZqcm51r)后可获得如下服务：

- 完整版包含「60+图像处理算法的原理和可独立运行的代码示例」文件
- 「图像处理工具集 - 完整版.exe」 程序
- 完整版支持批处理，一次处理多张图像
- 完整版支持所有60+种图像处理算法
- 一键完成60+种不同的处理算法图像处理
- 程序的所有源码

![gui](https://github.com/kangvcar/kkimage/blob/master/pics/app.gif?raw=true)

## :star2:实验结果

实验结果将会存放在该项目的**result**文件夹下，包括处理后图像的展示等。

## :clap:总结

在本项目中，我们使用Python OpenCV库，研究了多种图像处理算法，并实现了这些算法，得到了实验结果。本项目的实验结果可以作为未来研究的参考，并且可以用于更多的图像处理相关的研究

## 算法列表

1. 图像的坐标变换：平移变换、尺度变换、旋转变换、仿射变换
2. 灰度映射：求反、动态范围压缩、阶梯量化、阈值分割
3. 图像的算术运算：加法、平均法消除噪声、减法
4. 直方图修正：直方图均衡化、直方图规定化
5. 空域滤波：线性平滑滤波器、线性锐化滤波器、非线性平滑滤波器、非线性锐化滤波器
6. 频域图像增强：傅里叶变换和反变换
7. 高通和低通滤波器：理想滤波器、巴特沃斯滤波器，指数滤波器
8. 特殊高通滤波器：高频增强滤波器、高频提升滤波器
9. 带通带阻滤波器、同态滤波器
10. 空域噪声滤波器：均值滤波器、排序统计滤波器
11. 组合滤波器：混合滤波器、选择性滤波器
12. 无约束滤波器、有约束滤波器
13. 变长编码：哈夫曼编码、哥伦布编码、香农-法诺编码、算数编码、位平面编码
14. 预测编码：DPCM编码、余弦变换编码、小波变换编码
15. 图像分割：动态规划、单阈值分割
16. 典型分割：SUSAN边缘检测、主动轮廓、分水岭分割
17. 二值形态学：腐蚀、膨胀、开启、闭合
18. 基于二值形态学应用：噪声消除、目标检测、区域填充
19. 灰度形态学：腐蚀、膨胀、开启、闭合
20. 基于灰度形态学的应用：形态梯度、形态平滑、高帽变换、低帽变换
21. 算子：Sobel算子、Roberts算子、拉普拉斯算子、Canny算子、Prewitt算子、高斯拉普拉斯算子
