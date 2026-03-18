# -*- coding: utf-8 -*-
"""
检索业务服务引擎 (Service Layer / Facade Pattern)

这里使用了外观模式 (Facade Pattern)。引擎为复杂的底层子系统（特征提取、编码、重排序）
提供了一个统一的、简单的高层接口。UI 控制器只需要实例化这个引擎并调用其方法，
完全不需要了解底下是怎么做 K-Means 或者 RANSAC 的。
这也彻底实现了 UI 与业务逻辑的解耦。
"""

import os
import pickle
import numpy as np
from sklearn.preprocessing import normalize

# 导入我们之前写的模块
from config import FeatureMethod, CodebookMethod, EncodingMethod, ResortMethod
from feature_engine import LocalFeatureExtractor, GlobalFeatureExtractor, FeatureEncoder
from advanced_algorithms import QueryExpansion, ReRanker


class RetrievalEngine:
    def __init__(self):
        # --- 核心数据状态 ---
        self.train_path = None
        self.class_labels = []
        self.class_distribution = {}
        self.train_image_paths = []  # 格式:[(绝对路径, 类别), ...]
        self.all_descriptors = {}  # 缓存训练集所有图像的描述子 {路径: desc}
        self.train_codebook = None  # 视觉词典 (KMeans中心或GMM模型)
        self.train_encodings = None  # 训练集所有图像的最终编码矩阵
        self.idf = None  # TF-IDF 的 IDF 权重数组

        # --- 算法配置状态 ---
        self.feature_method = FeatureMethod.SIFT
        self.codebook_method = CodebookMethod.KMEANS
        self.encoding_method = EncodingMethod.VLAD
        self.codebook_count = 256
        self.enable_tfidf = False
        self.enable_qe = False
        self.resort_method = ResortMethod.NONE

    def save_model(self, file_path: str):
        """将引擎当前状态持久化保存 (序列化)"""
        data = {
            'train_path': self.train_path,
            'class_labels': self.class_labels,
            'class_distribution': self.class_distribution,
            'train_image_paths': self.train_image_paths,
            'train_codebook': self.train_codebook,
            'train_encodings': self.train_encodings,
            'idf': self.idf,
            'all_descriptors': self.all_descriptors,
            'feature_method': self.feature_method,
            'codebook_method': self.codebook_method,
            'encoding_method': self.encoding_method,
            'codebook_count': self.codebook_count,
            'enable_tfidf': self.enable_tfidf
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    def train(self, path: str, progress_callback=None):
        """一键训练流水线：扫描 -> 提取 -> 码本 -> 编码"""
        self.train_path = path
        self.class_labels = []
        self.class_distribution = {}
        self.train_image_paths = []

        # 1. 扫描目录获取类别
        sub_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        self.class_labels = sorted(sub_dirs)

        total_images = 0
        for cls in self.class_labels:
            cls_path = os.path.join(path, cls)
            valid_files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            self.class_distribution[cls] = len(valid_files)
            total_images += len(valid_files)
            self.train_image_paths.extend([(os.path.join(cls_path, f), cls) for f in valid_files])

        if total_images == 0: raise ValueError("训练集为空或未找到有效图片！")

        local_extractor = LocalFeatureExtractor(self.feature_method)
        self.all_descriptors = {}
        descriptors_list = []

        # 2. 提取特征
        if progress_callback: progress_callback(0, "正在提取训练集特征 (占30%)...")
        for idx, (img_path, _) in enumerate(self.train_image_paths):
            _, desc = local_extractor.extract(img_path)
            if desc is not None:
                self.all_descriptors[img_path] = desc
                descriptors_list.append(desc)
            if progress_callback: progress_callback(int((idx + 1) / total_images * 30), "正在提取训练集特征...")

        # 3. 生成视觉码本
        if progress_callback: progress_callback(30, "正在生成视觉码本 (较耗时)...")
        self.train_codebook = FeatureEncoder.create_codebook(descriptors_list, self.codebook_method,
                                                             self.codebook_count)

        # 4. 计算 TF-IDF
        self.idf = None
        if self.enable_tfidf:
            if progress_callback: progress_callback(60, "正在计算 TF-IDF 权重...")
            self.idf = np.zeros(self.codebook_count)
            for img_path, _ in self.train_image_paths:
                desc = self.all_descriptors.get(img_path)
                if desc is not None:
                    visual_words = FeatureEncoder.quantize_features(desc, self.train_codebook)
                    unique_words = np.unique(visual_words)
                    self.idf[unique_words] += 1
            N = len(self.train_image_paths)
            self.idf = np.log(N / (self.idf + 1e-6))

        # 5. 图像特征编码
        if progress_callback: progress_callback(70, "正在进行图像特征编码...")
        train_encodings = []
        for idx, (img_path, _) in enumerate(self.train_image_paths):
            desc = self.all_descriptors.get(img_path)
            encoding = FeatureEncoder.encode(desc, self.train_codebook, self.encoding_method, self.enable_tfidf,
                                             self.idf)
            train_encodings.append(encoding)
            if progress_callback: progress_callback(70 + int((idx + 1) / total_images * 30), "正在进行图像特征编码...")

        self.train_encodings = np.array([e for e in train_encodings if e is not None])
        if progress_callback: progress_callback(100, "训练完成！")

    def load_model(self, file_path: str):
        """从文件恢复引擎状态 (反序列化)"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # 【修复核心】兼容旧版本代码存下来的字典 Key 和 int 类型
        if 'feature_extraction_method' in data:
            self.feature_method = FeatureMethod(data['feature_extraction_method'])
            self.codebook_method = CodebookMethod(data['codebook_generate_method'])
            self.encoding_method = EncodingMethod(data['feature_encoding_method'])
            self.enable_tfidf = data.get('enable_tf_idf', False)
        else:
            self.feature_method = data['feature_method']
            self.codebook_method = data['codebook_method']
            self.encoding_method = data['encoding_method']
            self.enable_tfidf = data['enable_tfidf']

        self.train_path = data['train_path']
        self.class_labels = data['class_labels']
        self.class_distribution = data['class_distribution']
        self.train_image_paths = data['train_image_paths']
        self.train_codebook = data['train_codebook']
        self.train_encodings = data['train_encodings']
        self.idf = data['idf']
        self.all_descriptors = data['all_descriptors']
        self.codebook_count = data['codebook_count']

    def search(self, query_image_path: str, top_k: int) -> list:
        """
        核心检索流水线 (Pipeline)
        :param query_image_path: 查询图像路径
        :param top_k: 返回前 K 个结果
        :return: 排序好的结果列表，每个元素为 dict: {'label', 'path', 'dist', 'idx', 'score'(可选)}
        """
        if self.train_encodings is None:
            raise ValueError("引擎尚未训练或加载模型，无法进行检索！")

        # 1. 初始化提取器
        local_extractor = LocalFeatureExtractor(self.feature_method)
        global_extractor = GlobalFeatureExtractor()

        # 2. 提取查询图特征并编码
        query_kp, query_desc = local_extractor.extract(query_image_path)
        if query_desc is None:
            raise ValueError("无法提取测试图像的特征点。")

        query_encoding = FeatureEncoder.encode(
            query_desc, self.train_codebook, self.encoding_method,
            enable_tf_idf=self.enable_tfidf, idf=self.idf
        )

        # 强制 L2 归一化以计算距离
        train_enc_norm = normalize(self.train_encodings, norm='l2', axis=1)
        query_enc_norm = normalize(query_encoding.reshape(1, -1), norm='l2', axis=1)

        # 3. 初始距离计算 (欧氏距离)
        distances = np.linalg.norm(train_enc_norm - query_enc_norm, axis=1)

        # 4. 获取初始 Top-K (利用 argpartition 提升海量数据下的排序性能，时间复杂度 O(N))
        k = min(top_k, len(distances))
        nearest_indices = np.argpartition(distances, k)[:k]
        sorted_indices = nearest_indices[np.argsort(distances[nearest_indices])]

        # 组装初始结果列表
        results = []
        for idx in sorted_indices:
            path, label = self.train_image_paths[idx]
            results.append({
                'label': label,
                'path': path,
                'dist': distances[idx],
                'idx': idx
            })

        # 5. 扩展查询 (Query Expansion) - 提升召回率
        if self.enable_qe:
            avg_encoding = QueryExpansion.apply_qe(
                query_kp, query_desc, query_enc_norm, results,
                local_extractor, train_enc_norm, self.train_image_paths
            )
            # 使用融合后的新特征重新计算距离
            distances = np.linalg.norm(train_enc_norm - avg_encoding.reshape(1, -1), axis=1)
            nearest_indices = np.argpartition(distances, k)[:k]
            sorted_indices = nearest_indices[np.argsort(distances[nearest_indices])]

            results.clear()
            for idx in sorted_indices:
                path, label = self.train_image_paths[idx]
                results.append({
                    'label': label, 'path': path, 'dist': distances[idx], 'idx': idx
                })

        # 6. 重排序 (Re-ranking) - 提升精确度
        if self.resort_method == ResortMethod.GEOMETRIC:
            results = ReRanker.geometric_verification(results, query_kp, query_desc, local_extractor)
        elif self.resort_method == ResortMethod.LINEAR_COMBINATION:
            query_global_feat = global_extractor.extract(query_image_path)
            results = ReRanker.linear_combination(results, query_global_feat, global_extractor)

        return results