# -*- coding: utf-8 -*-
"""
项目全局配置文件
这里统一定义了系统所需的各种枚举常量。
使用枚举（Enum）代替魔法数字（Magic Numbers），提高了代码的可读性和类型安全性。
"""

from enum import Enum


class FeatureMethod(Enum):
    """特征提取算法枚举"""
    SIFT = 0
    ORB = 1


class CodebookMethod(Enum):
    """码本（视觉词典）生成算法枚举"""
    KMEANS = 0
    VQ = 1
    GMM = 2


class EncodingMethod(Enum):
    """特征编码算法枚举"""
    VLAD = 0
    BOF = 1
    FV = 2


class ResortMethod(Enum):
    """重排序算法枚举"""
    NONE = 0
    LINEAR_COMBINATION = 1  # 线性组合（基于颜色、形状、纹理等全局特征）
    GEOMETRIC = 2  # 几何验证（RANSAC 单应性矩阵验证）


# 默认算法配置
DEFAULT_FEATURE_METHOD = FeatureMethod.SIFT
DEFAULT_CODEBOOK_METHOD = CodebookMethod.KMEANS
DEFAULT_ENCODING_METHOD = EncodingMethod.VLAD
