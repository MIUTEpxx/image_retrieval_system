# -*- coding: utf-8 -*-
"""
高级检索算法模块 (扩展查询与重排序)

将复杂的算法逻辑从 UI 中剥离，不仅大幅提升了代码的可读性，
还使得这些算法变得“可独立测试 (Testable)”。
如果在实际工程中，算法工程师和前端开发人员就可以并行开发，互不干扰。
"""

import cv2
import numpy as np
from scipy import spatial


class QueryExpansion:
    """扩展查询 (QE) 模块"""

    @staticmethod
    def apply_qe(query_kp, query_desc, query_encoding, topk_results, feature_extractor, train_encodings,
                 train_image_paths):
        """
        基于几何验证（RANSAC）的扩展查询
        核心思想：将初次检索中，通过了几何验证的Top-K图像的特征，与原查询特征进行加权融合。
        """
        min_inliers = 10  # 几何验证所需的最小内点数
        validated_encodings = [query_encoding.ravel()]
        validated_sims = [1.0]  # 查询图自身相似度设为 1
        validated_paths = ["query_image"]

        bf = cv2.BFMatcher(cv2.NORM_L2)

        for result in topk_results:
            cand_path, cand_idx, cand_dist = result['path'], result['idx'], result['dist']
            cand_kp, cand_desc = feature_extractor.extract(cand_path)

            if cand_desc is None:
                continue

            # KNN 匹配并应用比率测试 (Lowe's ratio test)
            matches = bf.knnMatch(query_desc, cand_desc, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]

            # RANSAC 单应性矩阵验证
            if len(good) >= min_inliers:
                src_pts = np.float32([query_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([cand_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                # 内点数量大于阈值，认为几何一致性有效
                if mask is not None and mask.sum() > min_inliers:
                    validated_paths.append(cand_path)
                    validated_encodings.append(train_encodings[cand_idx].ravel())
                    validated_sims.append(1 - cand_dist)  # 距离转换为相似度

        # 动态调整K值（至少保留3个样本，或保留通过验证图像的 70%）
        K = max(3, int(len(validated_encodings) * 0.7))

        # Softmax 计算权重：使得相似度高的图像在特征融合时占比更大
        stacked_encodings = np.array(validated_encodings)[:K]
        stacked_sims = np.array(validated_sims)[:K]
        weights = np.exp(stacked_sims) / np.sum(np.exp(stacked_sims))

        # 加权平均融合特征
        avg_encoding = np.average(stacked_encodings, axis=0, weights=weights)

        # 返回前必须 L2 归一化，保持向量维度统一
        return avg_encoding / (np.linalg.norm(avg_encoding) + 1e-6)


class ReRanker:
    """重排序算法模块"""

    @staticmethod
    def linear_combination(initial_results, query_features, global_extractor):
        """
        基于颜色、形状、纹理全局特征的线性组合重排序
        """
        rerank_scores = []
        # 提取查询图特征
        test_color, test_shape, test_texture = query_features['color'], query_features['shape'], query_features[
            'texture']

        # 统计上下文相似度（同类图像占比，假设同类越多相关性越大）
        class_counter = {}
        for res in initial_results:
            label = res['label']
            class_counter[label] = class_counter.get(label, 0) + 1
        total_results = len(initial_results)

        for res in initial_results:
            label, path, base_dist = res['label'], res['path'], res['dist']
            base_sim = 1 / (1 + base_dist)

            # 提取候选图全局特征
            cand_feat = global_extractor.extract(path)

            # 颜色相似度（巴氏距离：越小越相似 -> 转换为越大越相似）
            color_sim = 1 - cv2.compareHist(test_color, cand_feat['color'], cv2.HISTCMP_BHATTACHARYYA)

            # 形状相似度（余弦相似度）
            shape_sim = 1 - spatial.distance.cosine(test_shape, cand_feat['shape'])

            # 纹理相似度（卡方检验：距离转相似度）
            texture_dist = cv2.compareHist(test_texture, cand_feat['texture'], cv2.HISTCMP_CHISQR)
            texture_sim = 1 / (1 + texture_dist)

            context_sim = class_counter[label] / total_results

            # 动态权重策略（重要思想：如果颜色差异极大，降低颜色权重，提升形状权重）
            weights = {'color': 0.25, 'shape': 0.35, 'texture': 0.25, 'context': 0.15}
            if color_sim < 0.3:
                weights['shape'] += weights['color'] * 0.5
                weights['color'] *= 0.5

            # 计算综合得分
            combined_score = (
                    weights['color'] * color_sim +
                    weights['shape'] * shape_sim +
                    weights['texture'] * texture_sim +
                    weights['context'] * context_sim +
                    0.2 * base_sim  # 基础相似度作为兜底
            )

            res['score'] = combined_score
            rerank_scores.append(res)

        # 按综合得分降序排列 (得分越大越靠前)
        return sorted(rerank_scores, key=lambda x: x['score'], reverse=True)

    @staticmethod
    def geometric_verification(initial_results, query_kp, query_desc, local_extractor):
        """
        基于几何验证的重排序
        通过验证的排在前面，未通过的排在后面；组内维持原有的距离顺序。
        """
        if not initial_results or query_desc is None:
            return initial_results

        min_inliers = 10
        bf = cv2.BFMatcher(cv2.NORM_L2)
        verified, unverified = [], []

        for res in initial_results:
            path = res['path']
            cand_kp, cand_desc = local_extractor.extract(path)

            if cand_desc is None:
                unverified.append(res)
                continue

            matches = bf.knnMatch(query_desc, cand_desc, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]

            if len(good) >= min_inliers:
                src_pts = np.float32([query_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([cand_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if M is not None and mask.sum() > min_inliers:
                    verified.append(res)
                else:
                    unverified.append(res)
            else:
                unverified.append(res)

        # 组内保持基于初始距离的升序排列
        verified.sort(key=lambda x: x['dist'])
        unverified.sort(key=lambda x: x['dist'])

        return verified + unverified