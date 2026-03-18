# -*- coding: utf-8 -*-
"""
图像特征提取与编码核心引擎
将“局部特征提取”、“全局特征提取”和“特征编码”拆分成独立的类。
每个类只干自己分内的事情，这符合面向对象的“单一职责原则 (SRP)”。
"""

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from skimage.feature import hog, local_binary_pattern

# 导入我们刚刚写的枚举配置
from config import FeatureMethod, CodebookMethod, EncodingMethod


class LocalFeatureExtractor:
    """
    局部特征提取器 (提取 SIFT 或 ORB 特征)
    主要用于基于词袋模型 (BoW) 的特征点提取
    """

    def __init__(self, method: FeatureMethod = FeatureMethod.SIFT):
        self.method = method

        # 策略模式 (Strategy Pattern) 的体现：根据配置动态选择底层算法
        if self.method == FeatureMethod.ORB:  # orb
            self.extractor = cv2.ORB_create(
                nfeatures=500,  # 控制特征点数量
                scaleFactor=1.2,  # 金字塔尺度因子
                nlevels=8,  # 金字塔层数
                edgeThreshold=15  # 边缘阈值
            )
        else:
            self.extractor = cv2.SIFT_create()  # sift

    def extract(self, image_path: str):
        """
        提取图像的特征点和描述子
        :param image_path: 图像物理路径
        :return: (keypoints, descriptors) 元组
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None

        keypoints, descriptors = self.extractor.detectAndCompute(img, None)

        # 统一数据类型：ORB 提取的描述子是 uint8 (二进制), SIFT 是 float32。
        # 为了后续聚类计算距离时不出错，统一转为 float32
        if descriptors is not None and self.method == FeatureMethod.ORB:
            descriptors = descriptors.astype(np.float32)

        return keypoints, descriptors


class GlobalFeatureExtractor:
    """
    全局特征提取器 (提取 颜色、形状、纹理)
    主要用于图像重排序 (Re-ranking) 阶段，补充局部特征缺失的宏观信息。
    """

    def __init__(self):
        # 颜色直方图的分箱数
        self.color_bins = 64
        # HOG (形状) 特征的参数配置
        self.hog_params = {
            'orientations': 12,
            'pixels_per_cell': (16, 16),
            'cells_per_block': (3, 3),
            'transform_sqrt': True
        }

    def extract(self, image_path: str) -> dict:
        """
        提取一张图的综合全局特征
        :return: 包含 'color', 'shape', 'texture' 的字典
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 强制统一图像尺寸！防止任意尺寸网图导致 HOG/LBP 维度不一致报错
        img = cv2.resize(img, (256, 256))

        features = {}

        # 1. 颜色特征（三维 LAB 颜色直方图）
        # 为什么用 LAB？因为 LAB 颜色空间比 RGB 更符合人类视觉感知。
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab = cv2.normalize(lab, None, 0, 255, cv2.NORM_MINMAX)
        hist = cv2.calcHist(
            images=[lab],
            channels=[0, 1, 2],
            mask=None,
            histSize=[8, 8, 8],  # 每个通道8个区间
            ranges=[0, 256, 0, 256, 0, 256]
        )
        features['color'] = cv2.normalize(hist, None).flatten()

        # 2. 形状特征（HOG - 梯度方向直方图）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hog_feat = hog(
            gray,
            orientations=self.hog_params['orientations'],
            pixels_per_cell=self.hog_params['pixels_per_cell'],
            cells_per_block=self.hog_params['cells_per_block'],
            channel_axis=None
        )
        # L2 归一化，防止不同尺寸图像导致数值差异过大
        features['shape'] = hog_feat.astype(np.float32) / np.linalg.norm(hog_feat + 1e-6)

        # 3. 纹理特征（LBP - 局部二值模式）
        # 统一模式 (uniform) 能够有效减少直方图维度，并具备旋转不变性
        lbp = local_binary_pattern(gray, P=24, R=3, method='uniform')
        hist_texture, _ = np.histogram(lbp, bins=256, range=(0, 256))
        hist_texture = hist_texture.astype(np.float32)
        features['texture'] = hist_texture / np.linalg.norm(hist_texture + 1e-6)

        return features


class FeatureEncoder:
    """
    特征编码器类
    包含：生成视觉码本、对描述子进行编码(BoF/VLAD/FV)等操作。
    """

    @staticmethod
    def create_codebook(descriptors_list: list, method: CodebookMethod, n_clusters: int):
        """
        根据提取到的所有图像局部描述子，聚类生成视觉词典（码本）
        """
        # 预处理：过滤无效的空描述子，并将列表纵向拼接成一个巨大的二维矩阵
        valid_descs = [d for d in descriptors_list if d is not None and len(d) > 0]
        if not valid_descs:
            raise ValueError("无有效特征描述符可用于生成码本")
        all_descs = np.vstack(valid_descs)

        if method == CodebookMethod.KMEANS:
            # 使用 MiniBatchKMeans 而不是标准 KMeans，这是为了应对海量图像特征点
            # 采用小批量梯度下降的思想，大幅降低内存消耗，提升速度
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024, max_iter=50, random_state=0)
            for desc in valid_descs:
                kmeans.partial_fit(desc)  # 增量/分批训练，防止撑爆内存
            return kmeans.cluster_centers_

        elif method == CodebookMethod.GMM:
            # 高斯混合模型 (GMM)，通常与 Fisher Vector (FV) 配合使用
            if len(all_descs) < n_clusters * 5:
                raise ValueError(f"GMM需要至少 {n_clusters * 5} 个样本")

            # 先用 KMeans 快速找到聚类中心，作为 GMM 的初始化参数，能有效防止 GMM 陷入局部极小值
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024).fit(all_descs)
            gmm = GaussianMixture(
                n_components=n_clusters,
                means_init=kmeans.cluster_centers_,
                covariance_type='diag',
                max_iter=100
            )
            gmm.fit(all_descs)
            return gmm

        elif method == CodebookMethod.VQ:
            # 向量量化 (Vector Quantization)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            normalized_descs = scaler.fit_transform(all_descs)

            vq = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', batch_size=1024, max_iter=100,
                                 compute_labels=False)
            chunk_size = 100000
            for i in range(0, len(normalized_descs), chunk_size):
                vq.partial_fit(all_descs[i:i + chunk_size])
            return vq.cluster_centers_

        else:
            raise ValueError("不支持的码本生成方法")

    @staticmethod
    def quantize_features(descriptors: np.ndarray, codebook: np.ndarray) -> np.ndarray:
        """
        将底层的描述子“量化”（映射）到最近的视觉单词上
        使用 KNN（K=1）实现
        """
        if descriptors is None or len(descriptors) == 0 or codebook is None:
            return None

        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(codebook)
        _, indices = nn.kneighbors(descriptors)
        return indices.flatten().astype(int)

    @staticmethod
    def encode(descriptors: np.ndarray, codebook, method: EncodingMethod, enable_tf_idf: bool = False,
               idf: np.ndarray = None) -> np.ndarray:
        """
        将一幅图像的多个局部特征描述子，编码为一个全局特征向量
        """
        if descriptors is None or len(descriptors) == 0:
            return None

        if method == EncodingMethod.BOF:
            # BoF (Bag of Features) 词袋模型编码
            indices = FeatureEncoder.quantize_features(descriptors, codebook)
            hist, _ = np.histogram(indices, bins=range(len(codebook) + 1), density=False)

            # TF：词频使用 log(1 + x) 进行平滑化，降低极端高频词（如大面积纯色背景产生的点）的干扰
            tf = np.log(1 + hist)

            if enable_tf_idf and idf is not None:
                hist = tf * idf
            else:
                hist = tf

            # 编码向量 L2 归一化：为了消除图像尺寸差异（特征点数量差异）带来的影响
            return hist / (np.linalg.norm(hist) + 1e-6)

        elif method == EncodingMethod.VLAD:
            # VLAD 编码：不仅统计数量（0阶统计量），还累加残差向量（1阶统计量）
            indices = FeatureEncoder.quantize_features(descriptors, codebook)
            vlad = np.zeros((len(codebook), descriptors.shape[1]))

            for k in range(len(codebook)):
                mask = (indices == k)
                if np.any(mask):
                    # 将属于第 k 个聚类中心的所有描述子减去聚类中心，得到残差并累加
                    vlad[k] = (descriptors[mask] - codebook[k]).sum(axis=0)

            # IDF 加权
            if enable_tf_idf and idf is not None:
                for k in range(len(codebook)):
                    vlad[k] *= idf[k]

            vlad = vlad.flatten()
            norm = np.linalg.norm(vlad)
            if norm > 0:
                vlad /= norm
            return vlad

        elif method == EncodingMethod.FV and isinstance(codebook, GaussianMixture):
            # FV (Fisher Vector) 编码：考虑了数据的概率分布（均值残差、方差残差）
            means, covs, weights = codebook.means_, codebook.covariances_, codebook.weights_
            post = codebook.predict_proba(descriptors)  # 计算后验概率

            d_means, d_sigmas = [], []
            for k in range(codebook.n_components):
                post_k = post[:, k]
                diff = descriptors - means[k]
                inv_sigma = 1 / np.sqrt(covs[k])

                # 均值的梯度
                grad_mean = post_k[:, None] * diff * inv_sigma
                d_means.append(grad_mean.sum(axis=0))

                # 方差的梯度
                grad_sigma = post_k[:, None] * (diff ** 2 * inv_sigma ** 3 - inv_sigma)
                d_sigmas.append(grad_sigma.sum(axis=0))

            fv = np.concatenate([np.concatenate(d_means), np.concatenate(d_sigmas)])
            return fv / (np.linalg.norm(fv) + 1e-6)

        else:
            raise ValueError("编码方法与码本不匹配，或暂不支持该编码算法")