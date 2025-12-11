# -*- coding: utf-8 -*-
"""特征对比/匹配模块"""

import os
import cv2
import numpy as np
from util.feature import getFeatureMap
from util.config import feature_dataset_path as fdp


def _imread_unicode(path, flags=cv2.IMREAD_UNCHANGED):
    """兼容包含中文路径的图像读取"""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def _parse_feature_dir(path):
    """
    解析特征目录路径，提取用户名和眼别
    目录格式: ./feature/name/L/ 或 ./feature/name/R/
    """
    normalized = os.path.normpath(path)
    parts = normalized.split(os.sep)
    
    if len(parts) < 2:
        return None, None

    eye = parts[-1]
    name = parts[-2]

    if eye not in ('L', 'R'):
        # 若缺少左右眼层级，则将最后一层视为姓名
        name = eye
        eye = ''

    return name, eye


def contrast(test_img, mode='swt', feature_dataset_path=fdp):
    """
    虹膜特征对比/识别
    
    :param test_img: 待识别的虹膜图像（灰度图）
    :param mode: 特征提取模式
    :param feature_dataset_path: 特征数据库路径
    :return: (name, eye, sorted_scores)
             - name: 匹配的用户名
             - eye: 匹配的眼别 ('L' 或 'R')
             - sorted_scores: 按距离排序的分数列表 [(key, score), ...]
    """
    score_dict = {}
    
    # 提取测试图像特征
    test_img_feature = getFeatureMap(test_img, mode)

    # 遍历特征数据库
    for path, _, file_list in os.walk(feature_dataset_path):
        if not file_list:
            continue

        name, eye = _parse_feature_dir(path)
        if not name:
            continue

        score_per_eye = 0
        valid_count = 0
        
        # 计算与该用户该眼所有样本的平均距离
        for img_name in file_list:
            img_path = os.path.join(path, img_name)
            img_feature = _imread_unicode(img_path, cv2.IMREAD_UNCHANGED)
            
            if img_feature is None:
                continue
            
            # 汉明距离（不同位的数量）
            distance = np.count_nonzero(test_img_feature != img_feature)
            score_per_eye += distance
            valid_count += 1

        if valid_count == 0:
            continue

        key = f"{name}-{eye or 'UNK'}"
        score_dict[key] = score_per_eye / valid_count

    if not score_dict:
        return None, None, []

    # 按分数排序（分数越低越匹配）
    sorted_list = sorted(score_dict.items(), key=lambda x: x[1])
    
    # 解析最佳匹配
    predict = sorted_list[0][0].split('-')
    name = predict[0]
    eye = predict[1] if len(predict) > 1 else ''
    
    return name, eye, sorted_list

