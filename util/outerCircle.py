# -*- coding: utf-8 -*-
"""外圆（虹膜边界）检测模块"""

import numpy as np
import cv2


def outerCircle(img, inner):
    """
    外圆检测（虹膜边界定位），需要先进行内圆检测
    :param img: cv2.imread() 读取的灰度图像 numpy.ndarray
    :param inner: 调用 innerCircle() 的返回值，瞳孔霍夫圆参数 [x, y, r]
    :return: 虹膜外边缘霍夫圆参数 numpy.ndarray [x, y, r]
    """
    # 裁剪图像区域（以瞳孔为中心）
    clip_img = img[(inner[1] - inner[2]): (inner[1] + inner[2]), :]
    
    # 图像增强
    clip_img = cv2.equalizeHist(clip_img)
    clip_img = cv2.GaussianBlur(clip_img, (9, 9), 0)
    clip_img = cv2.medianBlur(clip_img, 9)

    # 霍夫圆检测
    # 参数 minRadius 和 maxRadius 可根据内外圆半径比值调节
    circles = cv2.HoughCircles(
        clip_img, 
        cv2.HOUGH_GRADIENT, 
        2, 
        5,
        param1=30, 
        param2=20, 
        minRadius=int(inner[2] * 2.0), 
        maxRadius=int(inner[2] * 4)
    )
    
    circles = np.int16(np.around(circles))
    circles[0, :, 1] += inner[1] - inner[2]

    # 选择与内圆圆心距离最近的圆
    distance = np.zeros(len(circles[0]))
    for index, value in enumerate(circles[0, :]):
        distance[index] = np.linalg.norm(
            np.array((inner[0], inner[1])) - np.array((value[0], value[1]))
        )
    
    best_fit = circles[0][np.argmin(distance)]
    return best_fit

