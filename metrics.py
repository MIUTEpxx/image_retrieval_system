# -*- coding: utf-8 -*-
"""
评价指标计算模块

将数学运算与业务流程解耦。
以后无论是在本地测试、批量压测，还是做模型性能对比，
都可以直接调用这个独立的工具模块，保证了评价标准的一致性。
"""

import numpy as np


def evaluate_retrieval(results, true_label, total_relevant, k_value):
    """
    计算单次检索的各类指标 (Precision, Recall, AP, 以及 PR 曲线数据点)

    :param results: 检索出的结果列表，每个元素应包含 'label'
    :param true_label: 查询图像的真实类别
    :param total_relevant: 训练集中该类别的总图像数
    :param k_value: Top-K 的 K 值
    :return: 包含各项指标的字典
    """
    if not results or total_relevant == 0:
        return {
            'precision': 0.0, 'recall': 0.0, 'ap': 0.0,
            'precision_points': np.array([]), 'recall_points': np.array([])
        }

    correct_count = 0  # 截止当前位置，命中的相关文档数
    sum_precision = 0.0  # 累计精确度 (用于计算AP)

    precision_points = []  # 用于绘制 PR 曲线
    recall_points = []

    # 遍历 Top-K 结果（i从1开始）
    for i, res in enumerate(results, 1):
        is_correct = (res['label'] == true_label)
        if is_correct:
            correct_count += 1

        # 无论是否相关，都要计算该截断位置的 Precision
        current_precision = correct_count / i
        sum_precision += current_precision

        # 计算该截断位置的 Recall
        current_recall = correct_count / total_relevant

        precision_points.append(current_precision)
        recall_points.append(current_recall)

    # 计算最终指标
    final_precision = correct_count / len(results)
    final_recall = correct_count / total_relevant

    # AP (Average Precision) 计算标准：累加所有位置的精度，然后除以设定的 K 值
    ap = sum_precision / k_value

    return {
        'precision': final_precision,
        'recall': final_recall,
        'ap': ap,
        'correct_count': correct_count,
        'precision_points': np.array(precision_points),
        'recall_points': np.array(recall_points)
    }


def calculate_map(ap_history):
    """
    计算 mAP (Mean Average Precision)
    """
    if not ap_history:
        return 0.0
    # ap_history 列表中提取出所有的 AP 值求平均
    return np.mean([record['ap'] for record in ap_history])