# -*- coding: utf-8 -*-
"""虹膜归一化模块 - 将环形虹膜区域展开为矩形"""

import numpy as np


def normalize(image, x_iris, y_iris, r_iris, x_pupil, y_pupil, r_pupil,
              radpixels, angulardiv):
    """
    将环形虹膜区域归一化为矩形图像
    
    :param image: 输入虹膜图像
    :param x_iris, y_iris, r_iris: 虹膜边界圆参数
    :param x_pupil, y_pupil, r_pupil: 瞳孔边界圆参数
    :param radpixels: 径向分辨率（垂直维度）
    :param angulardiv: 角度分辨率（水平维度）
    :return: polar_array - 归一化后的虹膜区域
             polar_noise - 噪声区域标记
    """
    radiuspixels = radpixels + 2
    angledivisions = angulardiv - 1

    r = np.arange(radiuspixels)
    theta = np.linspace(0, 2 * np.pi, angledivisions + 1)

    # 计算瞳孔中心相对于虹膜中心的偏移
    ox = x_pupil - x_iris
    oy = y_pupil - y_iris

    if ox <= 0:
        sgn = -1
    elif ox > 0:
        sgn = 1

    if ox == 0 and oy > 0:
        sgn = 1

    a = np.ones(angledivisions + 1) * (ox ** 2 + oy ** 2)

    if ox == 0:
        phi = np.pi / 2
    else:
        phi = np.arctan(oy / ox)

    b = sgn * np.cos(np.pi - phi - theta)

    # 计算虹膜半径随角度的变化
    r = np.sqrt(a) * b + np.sqrt(a * b ** 2 - (a - r_iris ** 2))
    r = np.array([r - r_pupil])

    rmat = np.dot(np.ones([radiuspixels, 1]), r)
    rmat = rmat * np.dot(
        np.ones([angledivisions + 1, 1]),
        np.array([np.linspace(0, 1, radiuspixels)])
    ).transpose()
    rmat = rmat + r_pupil

    # 排除边界值
    rmat = rmat[1: radiuspixels - 1, :]

    # 计算笛卡尔坐标
    xcosmat = np.dot(np.ones([radiuspixels - 2, 1]), np.array([np.cos(theta)]))
    xsinmat = np.dot(np.ones([radiuspixels - 2, 1]), np.array([np.sin(theta)]))

    xo = rmat * xcosmat
    yo = rmat * xsinmat

    xo = x_pupil + xo
    xo = np.round(xo).astype(int)
    coords = np.where(xo >= image.shape[1])
    xo[coords] = image.shape[1] - 1
    coords = np.where(xo < 0)
    xo[coords] = 0

    yo = y_pupil - yo
    yo = np.round(yo).astype(int)
    coords = np.where(yo >= image.shape[0])
    yo[coords] = image.shape[0] - 1
    coords = np.where(yo < 0)
    yo[coords] = 0

    # 提取像素值
    polar_array = image[yo, xo]
    polar_array = polar_array / 255

    # 创建噪声数组
    polar_noise = np.zeros(polar_array.shape)
    coords = np.where(np.isnan(polar_array))
    polar_noise[coords] = 1

    # 绘制圆
    image[yo, xo] = 255
    x, y = circlecoords([x_iris, y_iris], r_iris, image.shape)
    image[y, x] = 255
    xp, yp = circlecoords([x_pupil, y_pupil], r_pupil, image.shape)
    image[yp, xp] = 255

    # 替换 NaN 值
    coords = np.where((np.isnan(polar_array)))
    polar_array2 = polar_array
    polar_array2[coords] = 0.5
    avg = np.sum(polar_array2) / (polar_array.shape[0] * polar_array.shape[1])
    polar_array[coords] = avg

    return polar_array, polar_noise.astype(bool)


def circlecoords(c, r, imgsize, nsides=600):
    """
    根据圆心和半径计算圆的坐标点
    
    :param c: 圆心 [x, y]
    :param r: 半径
    :param imgsize: 图像尺寸
    :param nsides: 圆的边数（默认 600）
    :return: x, y 坐标数组
    """
    a = np.linspace(0, 2 * np.pi, 2 * nsides + 1)
    xd = np.round(r * np.cos(a) + c[0])
    yd = np.round(r * np.sin(a) + c[1])

    # 限制在图像范围内
    xd2 = xd
    coords = np.where(xd >= imgsize[1])
    xd2[coords[0]] = imgsize[1] - 1
    coords = np.where(xd < 0)
    xd2[coords[0]] = 0

    yd2 = yd
    coords = np.where(yd >= imgsize[0])
    yd2[coords[0]] = imgsize[0] - 1
    coords = np.where(yd < 0)
    yd2[coords[0]] = 0

    x = np.round(xd2).astype(int)
    y = np.round(yd2).astype(int)
    return x, y

