# -*- coding: utf-8 -*-
"""特征提取模块 - 使用小波变换提取虹膜特征"""

import os
import cv2
import pywt
import numpy as np

from util.innerCircle import innerCircle
from util.outerCircle import outerCircle
from util.normalize import normalize
from util.config import feature_dataset_path as fdp
from util.config import dataset_path as dp

# 归一化后图像尺寸
HEIGHT = 40
WIDTH = 512


def _imread_unicode(path, flags=0):
    """兼容包含中文路径的图像读取"""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def _imwrite_unicode(path, img):
    """兼容包含中文路径的图像写入"""
    ext = os.path.splitext(path)[1]
    if not ext:
        ext = '.bmp'
        path = path + ext
    result, buffer = cv2.imencode(ext, img)
    if not result:
        raise IOError("图像编码失败，无法写入特征文件")
    with open(path, 'wb') as f:
        f.write(buffer.tobytes())


def generateFeatureDataset(feature_dataset_path=fdp, dataset_path=dp, mode='swt'):
    """
    生成特征数据库
    
    :param feature_dataset_path: 特征保存目录
    :param dataset_path: 原始图像目录，格式: ./photo/name/L/1.jpeg
    :param mode: 特征提取模式，'swt' 为静态小波变换
    """
    for path, _, file_list in os.walk(dataset_path):
        relative_dir = os.path.relpath(path, dataset_path)
        save_dir = os.path.join(feature_dataset_path, relative_dir) if relative_dir != '.' else feature_dataset_path
        os.makedirs(save_dir, exist_ok=True)

        for file_name in file_list:
            img_path = os.path.join(path, file_name)
            base_name, _ = os.path.splitext(file_name)
            save_path = os.path.join(save_dir, f"{base_name}.bmp")

            img = _imread_unicode(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"跳过无效图像: {img_path}")
                continue

            try:
                feature = getFeatureMap(img, mode)
                _imwrite_unicode(save_path, feature)
                print(f"特征生成: {save_path}")
            except Exception as e:
                print(f"特征提取失败 {img_path}: {e}")


def getFeatureMap(img, mode='swt'):
    """
    从虹膜图像提取特征
    
    :param img: 灰度虹膜图像
    :param mode: 'swt' 静态小波变换，'mallat' Mallat小波变换
    :return: 特征图 (320, 512) 的二值图像
    """
    # 定位内外圆
    inner = innerCircle(img)
    outer = outerCircle(img, inner)
    
    # 归一化
    polar_array, polar_noise = normalize(
        img, 
        outer[0], outer[1], outer[2],
        inner[0], inner[1], inner[2], 
        HEIGHT, WIDTH
    )
    
    # 特征提取
    if mode == 'swt':
        feature = swtFeatureMap(polar_array, wavelet='db3', level=7)
    else:
        feature = mallatFeatureMap(polar_array, wavelet='db3', level=7)
    
    return feature


def swtFeatureMap(normalized_img, wavelet='db3', level=7):
    """
    使用静态小波变换（SWT）提取特征
    
    :param normalized_img: 归一化后的虹膜图像
    :param wavelet: 小波基，默认 db3
    :param level: 分解层数，默认 7
    :return: 特征图 (320, 512)
    """
    swt_list = np.zeros((normalized_img.shape[0], level + 1, normalized_img.shape[1]), dtype=np.uint8)
    
    for index, value in enumerate(normalized_img):
        # 小波分解: cA7, cD7, cD6, cD5, cD4, cD3, cD2, cD1
        swt_coeffs = pywt.swt(data=value, wavelet=wavelet, level=level, trim_approx=True)
        
        # 二值化
        for coeff in swt_coeffs:
            coeff[coeff > 0] = 1
            coeff[coeff < 0] = 0
        
        swt_list[index] = np.array(swt_coeffs)
    
    swt_list = swt_list.swapaxes(0, 1)  # (8, 40, 512)
    feature = swt_list.reshape((-1, 512))  # (320, 512)
    
    return feature


def mallatFeatureMap(normalized_img, wavelet='db3', level=7):
    """
    使用 Mallat 小波变换提取特征（效果不如 SWT，备用）
    """
    coeff_list = np.zeros((normalized_img.shape[0], level + 1, normalized_img.shape[1]))
    
    for index, value in enumerate(normalized_img):
        coeffs = pywt.wavedec(data=value, wavelet=wavelet, level=level)
        for i, coeff in enumerate(coeffs):
            coeffs[i] = np.pad(
                np.array(coeff), 
                (0, normalized_img.shape[1] - len(coeff)), 
                'constant'
            )
        coeff_list[index] = np.array(coeffs)
    
    feature = coeff_list[4:5].reshape((-1, 512))
    return feature

