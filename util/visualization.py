# -*- coding: utf-8 -*-
"""可视化模块"""

import cv2


def displayCircle(img, outer_x, outer_y, outer_r, inner_x, inner_y, inner_r):
    """
    在图像上绘制内外圆检测结果
    :param img: 灰度图像
    :param outer_x, outer_y, outer_r: 外圆（虹膜）参数
    :param inner_x, inner_y, inner_r: 内圆（瞳孔）参数
    :return: 绘制了圆的彩色图像
    """
    # 转为彩色图像
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # 绘制内圆（瞳孔）- 绿色
    cv2.circle(cimg, (inner_x, inner_y), inner_r, (0, 255, 0), 1)
    
    # 绘制外圆（虹膜边界）- 绿色
    cv2.circle(cimg, (outer_x, outer_y), outer_r, (0, 255, 0), 1)
    
    # 绘制圆心点 - 红色
    cv2.circle(cimg, (inner_x, inner_y), 2, (0, 0, 255), 3)
    cv2.circle(cimg, (outer_x, outer_y), 2, (0, 0, 255), 3)
    
    return cimg

