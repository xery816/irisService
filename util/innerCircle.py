# -*- coding: utf-8 -*-
"""内圆（瞳孔）检测模块"""

import numpy as np
import cv2


def innerCircle(img):
    """
    内圆检测（瞳孔定位）
    :param img: cv2.imread() 读取的灰度图像 numpy.ndarray
    :return: 瞳孔霍夫圆参数 numpy.ndarray [x, y, r]
    """
    # 中值滤波去噪
    img = cv2.medianBlur(img, 11)
    
    # 二值化处理
    ret, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    # 霍夫圆检测
    circles = cv2.HoughCircles(
        img, 
        cv2.HOUGH_GRADIENT, 
        2, 
        5,
        param1=110, 
        param2=20, 
        minRadius=20, 
        maxRadius=130
    )
    
    circles = np.int16(np.around(circles))
    circles[0, :, 2] += 3  # 半径微调

    return circles[0][0]

