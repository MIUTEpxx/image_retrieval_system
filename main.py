import os
import time

import numpy as np
from scipy import spatial
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPen, QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QInputDialog
import sys

import IMGFP_pxx
import IMGFP_pxx as IMGFP

import feature_encoding_UI

import matplotlib

matplotlib.use('Agg')  # 避免 GUI 线程冲突
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QStackedWidget
from PyQt5.QtGui import QPixmap, QPainter
# from PyQt5.QtChart import QChart, QChartView, QLineSeries, QValueAxis
from PyQt5.QtCore import Qt

from skimage.feature import hog, local_binary_pattern  # 导入HOG和LBP函数
import numpy as np
import cv2





class main_window(QMainWindow, feature_encoding_UI.Ui_window_feature_encoding):
    def __init__(self):
        super().__init__()

        '''成员变量'''
        # 训练集路径
        self.train_path = None
        # 测试图片文件路径
        self.test_path = None
        # 测试图像
        self.test_image = None

        self.class_labels = []  # 存储所有类别名称
        self.class_distribution = {}  # 存储类别分布{类别名: 数量}
        # self.label_encodings = {}  # 存储带标签的编码{类别名: [编码数组]}
        self.current_test_label = ""  # 当前测试图片的真实标签（需要手动标注或通过目录结构获取）
        self.total_images = 0  # 图片总数

        self.test_mode = 0  # 0:单张 1:测试集
        self.test_set_container = {}  # 测试集容器 {类别名: [路径列表]}

        # 存储测试图片的编码
        self.test_encoding = None
        # 储存训练集图像的特征编码, 通过训练图像文件的路径来映射
        self.all_descriptors = {}
        # 存储训练集生成的码本
        self.train_codebook = None
        # 存储所有训练图的编码
        self.train_encodings = []
        # 存储元组列表[(绝对路径, 类别名), ...]
        self.train_image_paths = []
        # 训练集下的子目录集 作为类别
        self.classes = []

        # 当前特征提取算法
        self.feature_extraction_method = IMGFP.DEFAULT_pxx
        # 当前码本生成算法
        self.codebook_generate_method = IMGFP.DEFAULT_pxx
        # 当前特征编码算法
        self.feature_encoding_method = IMGFP.DEFAULT_pxx
        # 是否启用TF-IDF
        self.enable_tf_idf = False
        # 是否启用 重排序/重排序类型:1~2
        self.re_sort_method = 0
        # 是否启用 扩展查询
        self.enable_qe = False
        # 用于存储每个视觉单词的IDF值
        self.idf = None
        # 码本数量
        self.codebook_count = 256
        # KNN的K值
        self.k_value = 10
        # 特征匹配阈值
        self.feature_matching_threshold = 0.10
        # 训练集图像特征处理器
        self.train_processor = None
        # 测试图片图像特征处理器
        self.test_processor = None
        # 储存测试图像检测时产生的AP等指标值 用于绘制反应整体性能的pr曲线
        self.ap_history = []  # 存储元组 [(ap, precision_points, recall_points), ...]
        # 储存MAP值
        self.map_value = 0
        # 各个指标
        self.elapsed = 0
        self.precision = 0
        self.recall = 0

        '''UI组件'''
        # 初始化界面
        self.setupUi(self)
        # 初始化PR曲线画布
        self.pr_canvas_1 = None  # 用于存储当前画布对象
        self.pr_canvas_all = None
        '''初始化组件'''
        self.init_slot_connection()  # 初始化信号槽连接
        # self.init_default_values()  # 初始化控件默认值

    def init_slot_connection(self):
        """绑定所有控件的信号槽"""
        # 文件选择按钮
        self.btn_select_test_image.clicked.connect(self.select_test_image)  # 选择测试图片文件
        self.btn_select_train_dir.clicked.connect(self.select_train_dir)  # 选择训练集路径
        self.btn_detect_image.clicked.connect(self.start_detect)  # 开始检测
        self.btn_clear_result.clicked.connect(self.clear_result)  # 清除结果数据
        self.btn_save_training.clicked.connect(self.save_training_data)  # 保存训练结果
        self.btn_load_training.clicked.connect(self.load_training_data)  # 加载训练结果

        # 算法选择
        self.cmb_feature_extractor.currentIndexChanged.connect(self.update_feature_extractor)  # 特征提取算法
        self.cmb_encoding_method.currentIndexChanged.connect(self.update_encoding_method)  # 特征编码算法
        self.cmb_codebook_generate.currentIndexChanged.connect(self.update_codebook_generate)  # 码本生成算法
        self.cmb_re_sort.currentIndexChanged.connect(self.update_enable_re_sort)
        self.rbtn_tfidf.toggled.connect(self.update_enbal_if_idf)
        self.rbtn_qe.toggled.connect(self.update_enbal_qe)

        # self.rbtn_re_sort.toggled.connect(self.update_enable_re_sort)
        # 参数输入
        self.spb_codebook_size.valueChanged.connect(lambda v: setattr(self, 'codebook_count', v))
        self.spb_knn_k.valueChanged.connect(lambda v: setattr(self, 'k_value', v))
        # self.dspb_match_threshold.valueChanged.connect(lambda v: setattr(self, 'feature_matching_threshold', v))

    def start_detect(self):
        """执行图像检索"""
        if self.test_path is None or self.train_encodings is None:
            QMessageBox.warning(self, "错误", "请先选择测试图和训练集")
            return

        if self.cmb_test_format.currentIndex() == 0:  # 单张模式
            self.detect_image()  # 对单张图像文件进行检索

        else:
            # 遍历所有类别和图片
            elapsed_total = 0
            precision_total = 0
            recall_total = 0
            n = 0
            for class_name, img_paths in self.test_set_container.items():
                self.current_test_label = class_name
                if class_name not in self.class_labels:
                    QMessageBox.warning(self, "错误", f"未知类别{class_name}")
                    # self.current_test_label, ok = QInputDialog.getItem(
                    #     self, "选择类别", "请选择测试图像的类别：",
                    #     self.class_labels, 0, False
                    # )
                    continue
                for img_path in img_paths:
                    self.test_path = img_path
                    self.detect_image()  # 执行单次检测
                    elapsed_total += self.elapsed
                    precision_total += self.precision
                    recall_total += self.recall
                    n += 1

            self.lbl_detect_time.setText(f"平均耗时：{elapsed_total / n:.3f}秒")
            self.lbl_accuracy.setText(f"平均精度：{precision_total / n:.3%}")
            self.lbl_recall.setText(f"平均召回率：{recall_total / n:.3%}")
            self.lbl_ap.setText(f"A P值: XXX")
            self.lbl_map.setText(f"MAP值:{self.map_value:.3%}")
            self.lbl_result_num.setText(f"结果图像数:XXX")
            self.lbl_same_class_num.setText(f"同类图像数:XXX")
            self.lbl_class_num.setText(f"训练集中该类图像总数:XXX")
            self.lbl_pr_all.setText(f"整体Pr曲线 累计次数: {len(self.ap_history)}")

    def detect_image(self):
        # 开始计时
        start_time = time.time()  # 开始计时
        temp_descriptors = []  # 用于临时储存测试图和结果图的特征 用于后续扩展查询
        """测试图特征提取与编码"""
        processor = IMGFP.Processor(self.feature_extraction_method)  # 初始化特征提取器
        test_kp, test_desc = processor.extract_features(self.test_path)   # 提取测试图片特征描述子
        if test_desc is None:
            QMessageBox.warning(self, "错误", "无法提取测试图像特征")
            return

        temp_descriptors.append(test_desc)  # 保存测试图像的特征

        # 生成测试编码
        test_encoding = IMGFP.encode_features(
            test_desc,
            self.train_codebook,
            self.feature_encoding_method,
            idf=self.idf if self.enable_tf_idf else None,
            enable_tf_idf=self.enable_tf_idf
        )

        if test_encoding is None:
            QMessageBox.warning(self, "错误", "测试图像编码失败")
            return

        """相似度计算与结果排序"""
        # 强制归一化！
        train_encodings = normalize(self.train_encodings, norm='l2', axis=1)
        test_encoding = normalize(test_encoding.reshape(1, -1), norm='l2', axis=1)

        # if self.feature_encoding_method == IMGFP.BOF_pxx:
        #     # 使用余弦相似度（已归一化，等价于点积）
        #     similarities = np.dot(train_encodings, test_encoding.T).flatten()
        #     distances = 1 - similarities  # 余弦距离
        # else:  # 默认情况
        #     distances = np.linalg.norm(train_encodings - test_encoding, axis=1)  # 欧氏距离

        distances = np.linalg.norm(train_encodings - test_encoding, axis=1)  # 欧氏距离

        """获取Top-K结果"""
        k = min(self.k_value, len(distances))
        if k > len(self.train_encodings):
            k = len(self.train_encodings)  # 避免K值大于训练集图像数量

        # 使用np.argpartition高效地找到前k个最小距离的索引
        nearest_indices = np.argpartition(distances, k)[:k]
        # 对这k个索引进行排序，以确保结果是按距离递增的
        sorted_indices = nearest_indices[np.argsort(distances[nearest_indices])]  # 按相似度（距离）排序的图像索引

        # 构建结果列表
        results = []  # 格式 [(绝对路径,类别,相似度,idx)]
        for idx in sorted_indices:
            img_path, label = self.train_image_paths[idx]
            temp_descriptors.append(self.all_descriptors.get(img_path))  # 保存检索结果图像的特征
            results.append((label, img_path, distances[idx], idx))

        """是否进行扩展查询"""
        if self.enable_qe is True:
            # 检查有效性
            if not temp_descriptors:
                QMessageBox.warning(self, "错误", "扩展查询失败：无可用特征")
                return
            try:
                # ================== 阶段1：获取候选结果 ==================
                # 获取前K个结果的 索引
                topk_indices = sorted_indices[:self.k_value]
                # 获取前K个结果的 路径（包含测试图自身）
                topk_paths = [self.test_path]  # 首先将查询图像自身的路径加入
                for idx in topk_indices:
                    path, _ = self.train_image_paths[idx]
                    topk_paths.append(path)  # 添加搜索结果的图像路径

                # ================== 阶段2：几何验证 ,通过几何一致性验证，筛选出与查询图像在空间结构上一致的图像，排除误匹配==================
                # 初始化几何验证参数
                min_inliers = 10  # 进行几何验证时所需的最小内点数 内点(inliers)：符合几何变换关系的匹配点
                # 如果匹配点数量超过阈值（min_inliers=10），则认为候选图片A是有效的,
                # 否则 如果匹配点经过单应性矩阵验证后，内点数量少于这个阈值，则认为几何一致性不足

                validated_paths = [self.test_path]  # 始终包含原查询图

                # 提取测试图像的特征点（keypoints）和描述子（descriptors）
                query_kp, query_desc = test_kp, test_desc  # 直接使用已提取的测试图特征
                if query_desc is not None:
                    # 创建BFMatcher
                    bf = cv2.BFMatcher(cv2.NORM_L2)  # 暴力匹配器，用于快速匹配特征点, cv2.NORM_L2 表示使用 L2 范数（欧氏距离）来计算描述子之间的距离
                    for cand_path in topk_paths[1:]:  # [1:]跳过查询图自身
                        # 提取当前候选图像的特征点和描述子。
                        cand_kp, cand_desc = processor.extract_features(cand_path)
                        if cand_desc is None:
                            continue

                        # 对查询图像和候选图像的描述子进行 K-Nearest Neighbors (KNN) 匹配, k=2 表示为每个查询描述子找到两个最近的匹配
                        matches = bf.knnMatch(query_desc, cand_desc, k=2)

                        # 应用比率测试 过滤掉不可靠的匹配（只保留最独特的匹配）
                        good = []
                        for m, n in matches:
                            # 匹配点之间的距离比是否大于某个阈值（这里是 0.75!）
                            if m.distance < 0.75 * n.distance:
                                good.append(m)  # 认为该匹配点是可靠的，否则认为是噪声或歧义匹配，将其舍弃

                        if len(good) < min_inliers:
                            continue  # 可靠的匹配点数量少于 min_inliers，则认为两张图像的特征匹配不足，跳过当前候选图像

                        # 进行单应性矩阵验证 使用RANSAC算法估计两张图片之间的几何变换关系 (就像看两张图是不是同一个物体在不同角度、不同距离拍的 如果是，就保留；如果不是，就排除)
                        src_pts = np.float32([query_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                        dst_pts = np.float32([cand_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                        # 使用 RANSAC 算法（Random Sample Consensus）估计两组点之间的单应性矩阵
                        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        # mask：RANSAC 算法返回一个掩码，其中 mask[i] 为 1 表示第 i 个匹配点是“内点”（inlier），即符合估计的单应性变换；为 0 表示是“外点”

                        if mask.sum() > min_inliers:  # 如果内点数量大于 min_inliers，则认为查询图像与候选图像之间存在有效的几何一致性
                            validated_paths.append(cand_path)

                # ================== 阶段3：权重计算 ==================
                # 获取验证通过的编码向量
                validated_encodings = [test_encoding.ravel()]  # 展平为1D数组 初始值为测试图的编码
                validated_sims = [1.0]  # 查询图自身相似度设为1
                for path in validated_paths[1:]:
                    # 添加安全检查
                    try:
                        # 通过路径查找该图像在训练集中的索引
                        idx_in_train = next(i for i, (p, _) in enumerate(self.train_image_paths) if p == path)
                        encoding = self.train_encodings[idx_in_train]  # 获取对应图像的特征编码

                        # 统一维度处理 确保所有编码都是一维的，以进行后续的加权平均
                        if encoding.ndim == 2:
                            encoding = encoding.ravel()  # 展平为1D
                        elif encoding.ndim != 1:
                            raise ValueError(f"无效编码维度：{encoding.shape}")

                        validated_encodings.append(encoding)  # 将通过验证的图像编码加入列表
                        validated_sims.append(1 - distances[idx_in_train])  # 将该图像与查询图像的相似度（1减去距离）加入列表。距离越小，相似度越大
                    except (StopIteration, IndexError):
                        print(f"警告：未找到路径 {path} 的编码")
                        continue

                # 动态调整K值（至少保留3个样本 或 使用通过几何验证的图像数量的 70%）
                K = max(3, int(len(validated_encodings) * 0.7))  # 保留70%的验证结果

                # 计算权重（softmax归一化, 将相似度转换为 0 到 1 之间的概率分布，所有权重之和为 1）
                weights = np.exp(validated_sims) / np.sum(np.exp(validated_sims))  # 这意味着相似度越高的图像，其对应的权重越大，在后续的加权平均中贡献越大。

                # ================== 阶段4：加权特征融合 ==================
                # 把测试图片的特征和 阶段3 筛选出来的、打好权重的相关图片的特征加权平均起来
                # 想象成把这些图片的特征“融合”成一个新的、更精准的“查询特征向量” 权重越大的图片，其特征在融合中贡献越大

                # 转换前检查所有编码长度
                dim_sizes = {e.shape[0] for e in validated_encodings}
                if len(dim_sizes) != 1:  # 确保所有参与融合的编码都具有相同的维度，防止因维度不一致导致的错误
                    raise ValueError(f"编码维度不一致：发现不同长度 {dim_sizes}")
                stacked_encodings = np.array(validated_encodings)[:K]  # 表转换为 NumPy 数组，并根据动态调整后的 K 值截取前 K 个编码
                stacked_weights = weights[:K]  # 截取前 K 个权重

                # 加权平均
                avg_encoding = np.average(stacked_encodings, axis=0, weights=stacked_weights)
                avg_encoding = normalize(avg_encoding.reshape(1, -1), norm='l2', axis=1)

                # ================== 阶段5：重计算距离并再次进行图像检索 ==================
                distances = np.linalg.norm(self.train_encodings - avg_encoding, axis=1)

                # 扩展查询 再次获取前K个结果
                nearest_indices = np.argpartition(distances, k)[:k]
                sorted_indices = nearest_indices[np.argsort(distances[nearest_indices])]

                # 构建结果列表
                results.clear()
                for idx in sorted_indices:
                    img_path, label = self.train_image_paths[idx]
                    results.append((label, img_path, distances[idx], idx))

            except Exception as e:
                QMessageBox.warning(self, "错误", f"扩展查询失败: {str(e)}")
                print(f"扩展查询失败: {str(e)}")
                return

        """是否重排序"""
        if self.re_sort_method == IMGFP.GEOMETRIC_pxx:
            # 基于几何验证的重排序
            if not results:
                QMessageBox.warning(self.ui, "警告", "无结果可重排序。")
                self.display_results(results, time.time() - start_time)
                return

            # 准备几何验证所需的特征提取器和描述子
            query_kp_for_resort = test_kp
            query_desc_for_resort = test_desc

            # 调用重排序方法
            # 传递当前results和用于几何验证的查询特征
            results = self.reorder_results(results, query_kp_for_resort, query_desc_for_resort, processor)

        # 更改格式以适应后续代码 # 格式 [(绝对路径,类别,相似度)]
        results_ = []
        for result in results:
            label, path, dist, _ = result
            results_.append((label, path, dist))

        results = results_

        if self.re_sort_method == IMGFP.LC_pxx:
            # 启用基于线性组合的重排序
            test_features = processor.extract_global_features(self.test_path)  # 提取测试图像的颜色/形状/纹理特征
            # 执行重排序
            final_results = self.linear_reranking(processor, results, test_features)
            results = final_results

        # 显示结果图像
        self.show_result(results)

        """计算指标"""
        elapsed = time.time() - start_time  # 时间
        true_label = self.current_test_label  # 测试图像的类别
        total_relevant = sum(1 for _, label in self.train_image_paths if label == true_label)  # 总相关文档数（训练集中该类别的样本总数）

        # 生成相关标记数组
        relevant = [1 if label == true_label else 0 for label, _, _ in results]  # 同类为1, 否则为零 形成数组

        # 精确值和召回率计算
        precision = sum(relevant) / len(relevant)  # 精度 = 结果中同类的数量/结果总数
        recall = sum(relevant) / total_relevant if total_relevant > 0 else 0  # 结果中同类数/此类总数

        # 计算平均精度AP与PR曲线数据

        relevant_found = 0  # 已找到的相关文档数量

        # 初始化存储 PR 数据的列表
        precision_points = []  # 存储每个位置的精度（用于绘制PR曲线）
        recall_points = []  # 存储每个位置的召回率（用于绘制PR曲线）

        # 遍历排序后的结果（i从1开始，表示第1个位置）
        correct = 0  # 记录截止到当前已检索到的正确结果（相关文档）数量
        sum_precision = 0.0  # 累加所有位置的精度（无论是否相关）

        for i, (label, _, _) in enumerate(results, 1):  # i从1开始到k_value
            # 判断是否相关并更新correct计数
            if label == true_label:
                correct += 1
                relevant_found += 1

            # 无论是否相关，都计算当前精度并累加
            current_precision = correct / i
            sum_precision += current_precision

            # 当前召回率计算, 储存召回率与对应精确度, 作为PR曲线的一点
            current_recall = relevant_found / total_relevant if total_relevant > 0 else 0
            precision_points.append(current_precision)
            recall_points.append(current_recall)

        # 计算AP：所有位置的精度之和 / 结果总数（k_value）
        AP = sum_precision / self.k_value

        # 存储本次检测的完整PR数据
        self.ap_history.append((
            AP,  # ap
            np.array(precision_points),  # 精确度
            np.array(recall_points)  # 召回率
        ))
        # 计算当前mAP（平均所有AP）
        self.map_value = np.mean([ap for ap, _, _ in self.ap_history])  # 计算MAP
        # self.map_value = sum(self.ap_history) / len(self.ap_history)

        # 绘制单次测试的pr曲线
        self.plot_pr_curve(recall_points, precision_points, AP, self.widget_pr_chart_1)
        # 绘制整体性能曲线
        self.update_overall_pr_curve()
        # 显示各个指标
        self.elapsed = elapsed
        self.precision = precision
        self.recall = recall

        self.lbl_detect_time.setText(f"耗时：{elapsed:.3f}秒")
        self.lbl_accuracy.setText(f"精度：{precision:.3%}")
        self.lbl_recall.setText(f"召回率：{recall:.3%}")
        self.lbl_ap.setText(f"A P值: {AP:.3%}")
        self.lbl_map.setText(f"MAP值:{self.map_value:.3%}")
        self.lbl_result_num.setText(f"结果图像数:{self.k_value}")
        self.lbl_same_class_num.setText(f"同类图像数:{sum(relevant)}")
        self.lbl_class_num.setText(f"训练集中该类图像总数:{self.class_distribution[self.current_test_label]}")
        self.lbl_pr_all.setText(f"整体Pr曲线 累计次数: {len(self.ap_history)}")

    def linear_reranking(self, processor, initial_results, test_features,
                         weights={'color': 0.25, 'shape': 0.35, 'texture': 0.25, 'context': 0.15}):
        """基于 颜色/形状/纹理/类别 的线性组合重排序"""
        rerank_scores = []  # 用于存储每个候选图像经过重排序后计算得到的综合得分及相关信息 最终这个列表将被用来进行排序

        #  将查询图像的颜色、形状、纹理特征从 test_features 字典中分别提取出来，方便后续在循环中直接使用，避免重复通过字典键访问
        test_color = test_features['color']
        test_shape = test_features['shape']
        test_texture = test_features['texture']

        # 计算上下文相似度（同类图像占比）
        #这部分是为了计算“上下文相似度”
        # 一个简单的假设是，如果初始检索结果中某个类别的图像数量越多，那么该类别可能与查询图像更相关
        # 这里的“上下文”指的是初始检索结果集中图像类别的分布
        class_counter = {}  # 用于统计 initial_results 中每个图像类别（label）出现的次数
        for label, _, _ in initial_results:
            class_counter[label] = class_counter.get(label, 0) + 1
        total = len(initial_results)

        for idx, (label, path, base_dist) in enumerate(initial_results):
            # 基础相似度转换
            base_sim = 1 / (1 + base_dist)  # 将欧式距离转换为相似度

            # 提取候选图全局特征
            candidate_features = processor.extract_global_features(path)  # 提取颜色、形状和纹理特征

            # 颜色相似度（改进巴氏距离,越小越相似,通过 "1 - 距离" 的方式将其转换为相似度, 值越大表示颜色越相似）
            color_sim = 1 - cv2.compareHist(test_color, candidate_features['color'],
                                            cv2.HISTCMP_BHATTACHARYYA)

            # 形状相似度（HOG余弦相似度, 计算方式同上）
            shape_sim = 1 - spatial.distance.cosine(test_shape, candidate_features['shape'])

            # 纹理相似度（LBP卡方检验）
            texture_sim = cv2.compareHist(test_texture.astype(np.float32),
                                          candidate_features['texture'].astype(np.float32),
                                          cv2.HISTCMP_CHISQR)
            # 卡方距离的值可以非常大（越大越不相似）, 这里采用 1 / (1 + 距离) 的方式将其转换为相似度
            texture_sim = 1 / (1 + texture_sim)  # 转换到[0,1]

            # 上下文相似度（同类结果占比）
            context_sim = class_counter[label] / total

            # 动态权重调整（如果颜色差异过大则降低颜色权重）
            if color_sim < 0.3:
                adj_weights = weights.copy()
                adj_weights['color'] *= 0.5
                adj_weights['shape'] += weights['color'] * 0.5
            else:
                adj_weights = weights

            # 综合得分
            combined_score = (
                    adj_weights['color'] * color_sim +
                    adj_weights['shape'] * shape_sim +
                    adj_weights['texture'] * texture_sim +
                    adj_weights['context'] * context_sim +
                    0.2 * base_sim  # 保留基础相似度影响
            )

            rerank_scores.append((label, path, combined_score))

        # 按综合得分降序排序
        rerank_scores.sort(key=lambda x: -x[2])
        return rerank_scores

    def reorder_results(self, results, query_kp, query_desc, feature_processor):
        """
        基于几何验证的重排序方法。
        将通过几何验证的图像排序到结果队列的前面，否则排序到后面。
        在通过和未通过的组内，保持原始距离排序。

        results: 当前的检索结果列表 [(label, path, distance, original_idx), ...]
        query_kp: 查询图像的关键点
        query_desc: 查询图像的描述子
        feature_processor: 用于提取候选图像特征的 Processor 实例
        """
        if not results or query_desc is None:
            return results

        min_inliers = 10  # 最小内点数阈值，与扩展查询中保持一致

        geometric_verified_results = []
        non_geometric_verified_results = []

        # 创建BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_L2)  # 暴力匹配器，用于快速匹配特征点

        for result_item in results:
            label, img_path, dist, original_idx = result_item

            try:
                # 提取候选图特征
                cand_kp, cand_desc = feature_processor.extract_features(img_path)
                if cand_desc is None:
                    non_geometric_verified_results.append(result_item)
                    continue

                # 特征匹配
                matches = bf.knnMatch(query_desc, cand_desc, k=2)

                # 应用比率测试
                good = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append(m)

                if len(good) < min_inliers:
                    non_geometric_verified_results.append(result_item)
                    continue

                # 进行单应性矩阵验证
                if len(good) < 4:  # RANSAC至少需要4个点
                    non_geometric_verified_results.append(result_item)
                    continue

                src_pts = np.float32([query_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([cand_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if M is not None and mask.sum() > min_inliers:  # 通过几何验证
                    geometric_verified_results.append(result_item)
                else:
                    non_geometric_verified_results.append(result_item)

            except Exception as e:
                print(f"几何重排序处理图片 {img_path} 时发生错误: {str(e)}")
                non_geometric_verified_results.append(result_item)  # 出错的也放入未通过组

        # 在各自的组内，保持原始距离排序
        geometric_verified_results.sort(key=lambda x: x[2])
        non_geometric_verified_results.sort(key=lambda x: x[2])

        # 合并结果：通过几何验证的在前，未通过的在后
        reordered_results = []
        reordered_results.extend(geometric_verified_results)
        reordered_results.extend(non_geometric_verified_results)

        return reordered_results

    def show_result(self, results):
        """显示检测结果图像和信息"""
        self.list_results.clear()

        item_size = QtCore.QSize(200, 150)  # 统一项目尺寸
        self.list_results.setGridSize(item_size)

        for result in results:
            # 解包不同长度的结果
            if len(result) == 4:
                label, path, dist, _ = result
            else:
                label, path, dist = result
            # 创建列表项和对应的widget
            item = QtWidgets.QListWidgetItem()
            widget = self._create_result_item(label, path, dist)

            # 设置项目尺寸
            item.setSizeHint(item_size)

            # 正确添加项目到列表
            self.list_results.addItem(item)
            self.list_results.setItemWidget(item, widget)

    def _create_result_item(self, label, path, dist):
        """创建单个结果项（紧凑版）"""
        widget = QtWidgets.QGroupBox()
        layout = QtWidgets.QVBoxLayout(widget)

        # 缩略图（更小尺寸）
        thumbnail = QtWidgets.QLabel()
        pixmap = QPixmap(path).scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        thumbnail.setPixmap(pixmap)
        thumbnail.setAlignment(Qt.AlignCenter)
        layout.addWidget(thumbnail)

        similar_info = f"相似度:{(1 - dist):.1%}" if self.feature_encoding_method == IMGFP.FV_pxx else f"相似度:{(1 / (1 + dist)):.1%}"
        # 信息区域
        info = QtWidgets.QLabel(
            f"<b>{label}</b><br>"
            f"{os.path.basename(path)}<br>"
            + similar_info
        )
        info.setAlignment(Qt.AlignCenter)
        layout.addWidget(info)

        # 紧凑样式
        widget.setStyleSheet("""
            QGroupBox {
                border: 1px solid #ddd;
                border-radius: 4px;
                margin: 2px;
                padding: 5px;
            }
            QLabel { font-size: 12px; }
        """)
        return widget

    def plot_pr_curve(self, recall, precision, ap, target_widget):
        """在指定widget中绘制PR曲线"""
        # 清除旧图表
        if target_widget.layout() is not None:
            while target_widget.layout().count():
                child = target_widget.layout().takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
        else:
            target_widget.setLayout(QtWidgets.QVBoxLayout())

        # 创建新图表
        fig = Figure(figsize=(5, 4), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        # 绘制曲线
        ax.step(recall, precision, where='post', label=f'AP = {ap:.3f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_ylim([0.0, 1.05])
        ax.set_xlim([0.0, 1.05])
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc='lower left')

        # 嵌入到目标widget
        target_widget.layout().addWidget(canvas)

        # 缓存画布对象
        if target_widget == self.widget_pr_chart_1:
            self.pr_canvas_1 = canvas
        else:
            self.pr_canvas_all = canvas

    def update_overall_pr_curve(self):
        """在指定widget中更新综合PR曲线"""
        if not self.ap_history:
            return

        # 确定所有记录的最小召回范围
        min_recalls = [np.min(recalls) for ap, precisions, recalls in self.ap_history if len(recalls) > 0]
        if not min_recalls:
            return

        # 从最大最小召回值开始插值（保证所有记录在此范围内有数据）
        max_min_recall = max(min_recalls)
        recall_levels = np.linspace(max_min_recall, 1, 100)

        interp_precisions = []

        for ap, precisions, recalls in self.ap_history:
            # 确保召回序列单调递增
            sorted_indices = np.argsort(recalls)
            recalls_sorted = recalls[sorted_indices]
            precisions_sorted = precisions[sorted_indices]

            # 获取当前记录的最小召回点精度
            if len(recalls_sorted) == 0:
                continue
            min_precision = precisions_sorted[0]
            max_recall = recalls_sorted[-1]  # 当前记录的最大召回值

            # 插值时使用当前记录首点精度作为左边界
            interp_prec = np.interp(
                recall_levels,
                recalls_sorted,
                precisions_sorted,
                left=min_precision,  # 使用当前记录的最小召回点精度
                right=precisions_sorted[-1] if len(precisions_sorted) > 0 else min_precision  # 使用当前记录的最大召回点精度
            )
            # 对于超出当前记录最大召回值的点，保持最大召回点的精度值
            # 这样可以避免曲线垂直下降到0
            interp_prec[recall_levels > max_recall] = precisions_sorted[-1] if (
                    len(precisions_sorted) > 0) else min_precision

            interp_precisions.append(interp_prec)

        # 计算平均精度时忽略全零数据
        valid_precisions = [p for p in interp_precisions if np.any(p > 0)]
        if not valid_precisions:
            return

        avg_precision = np.mean(valid_precisions, axis=0)  # 平均值计算
        current_map = self.map_value

        # 裁剪有效范围（从第一个非零点开始）
        first_valid = np.argmax(avg_precision > 0)
        valid_recalls = recall_levels[first_valid:]
        valid_precision = avg_precision[first_valid:]

        self.plot_pr_curve(
            valid_recalls,
            valid_precision,
            current_map,
            self.widget_pr_chart_all
        )

    def select_test_image(self):
        """选择测试图片 or 测试集"""
        if self.class_labels is None or self.class_labels == []:
            QMessageBox.warning(self, "注意", "请先选择训练集")
            return
        if self.cmb_test_format.currentIndex() == 0:  # 单张模式
            path, _ = QFileDialog.getOpenFileName(
                self, "选择测试图片", "",
                "Image Files (*.jpg *.jpeg *.png *.bmp)"
            )
            if path:
                # 自动推断标签（测试图像在类子目录中）
                parent_dir = os.path.basename(os.path.dirname(path))
                if parent_dir in self.class_labels:
                    self.current_test_label = parent_dir
                else:
                    # 手动选择标签
                    self.current_test_label, ok = QInputDialog.getItem(
                        self, "选择类别", "请选择测试图像的类别：",
                        self.class_labels, 0, False
                    )
                    if not ok:
                        return
                self.label_test_img.setText(f"测试图像 类别:{self.current_test_label}")
                self.lbl_class_num.setText(f"训练集中该类图像总数:{self.class_distribution[self.current_test_label]}")
                self.test_path = path
                self.lbl_test_image_path.setText(path)
                self.show_test_image(path)

        else:  # 测试集模式
            test_dir = QFileDialog.getExistingDirectory(
                self, "选择测试集目录（需包含类别子目录）"
            )
            if test_dir:
                self.test_set_container.clear()  # 清空旧数据

                # 遍历子目录
                for class_name in os.listdir(test_dir):
                    class_path = os.path.join(test_dir, class_name)
                    if os.path.isdir(class_path):
                        img_paths = []
                        # 收集有效图片路径
                        for f in os.listdir(class_path):
                            if f.split('.')[-1].lower() in {'jpg', 'jpeg', 'png', 'bmp'}:
                                img_paths.append(os.path.join(class_path, f))

                        if img_paths:
                            self.test_set_container[class_name] = img_paths

                # 有效性检查
                if not self.test_set_container:
                    QMessageBox.warning(self, "错误", "测试目录无有效类别子目录")
                    self.self.test_path = None
                    return

                self.test_path = test_dir
                self.lbl_test_image_path.setText(test_dir)

    def show_test_image(self, image_path):
        """显示测试图片"""
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            # 按比例缩放适应显示区域
            scaled_pixmap = pixmap.scaled(
                self.lbl_test_image.width(),
                self.lbl_test_image.height(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            self.lbl_test_image.setPixmap(scaled_pixmap)
            self.test_image = pixmap  # 保存原始图像对象

    def select_train_dir(self):
        """选择训练集目录"""
        path = QFileDialog.getExistingDirectory(
            self,
            "选择训练集目录（需包含类别子目录）",
            "",
            QFileDialog.ShowDirsOnly
        )

        if not path:
            return

        # 初始化数据结构
        self.class_labels.clear()
        self.class_distribution.clear()
        self.train_image_paths.clear()

        # 显示等待光标
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            # 获取子目录作为类别
            sub_dirs = [d for d in os.listdir(path)
                        if os.path.isdir(os.path.join(path, d))]

            # 验证至少有两个类别
            if len(sub_dirs) < 2:
                QMessageBox.critical(self, "错误",
                                     "训练目录必须包含至少两个类别子文件夹！")
                return

            # 保存训练集主路径
            self.train_path = path

            # 储存所有类别(子目录名)
            self.class_labels = sorted(sub_dirs)
            # 图片总数
            total_images = 0

            # 遍历每个类别目录
            for class_name in self.class_labels:
                class_path = os.path.join(path, class_name)
                valid_files = [
                    f for f in os.listdir(class_path)
                    if os.path.splitext(f)[1].lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
                ]

                # 记录到成员变量
                self.class_distribution[class_name] = len(valid_files)
                total_images += len(valid_files)

                # 构建路径元组
                self.train_image_paths.extend(
                    [(os.path.join(class_path, f), class_name)
                     for f in valid_files]
                )

            # 更新当前训练集图片总数
            self.total_images = total_images
            # 更新UI显示
            self._update_train_ui(total_images)

            # 选择训练集后,就自动开始处理训练集
            self.train_image_encoding()

            # 重置各指标值
            self.clear_result()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"读取训练集失败: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()  # 恢复光标

    def _update_train_ui(self, total):
        """更新训练集信息显示"""
        info_lines = [
            f"总样本数: {total}",
            f"类别数: {len(self.class_labels)}",
            "类别分布:"
        ]

        # 添加每个类别的统计信息
        for cls, count in self.class_distribution.items():
            info_lines.append(f"   • {cls}: {count}张")

        # 设置文本框显示
        self.lbl_train_dir_info.setText("\n".join(info_lines))

        # 更新路径显示（截断长路径）
        # display_path = f"{os.path.split(self.train_path)[0]}/.../{os.path.basename(self.train_path)}"
        self.lbl_train_dir_path.setText(f"训练集路径: {self.train_path}")

        # 启用开始检测按钮
        self.btn_detect_image.setEnabled(True)

    def train_image_encoding(self):
        """
        1、用训练集的图像用k-means或VQ等算法生成字典（码本）
        2、用BoF, VLAD,FV等算法对图像特征进行编码
        3、保存训练集中所有图像的特征编码
        4、加入了TF-IDF
        """

        """处理训练集生成码本和编码"""
        if not self.train_path:
            return

        # 收集所有训练图像的特征描述子
        # QApplication.setOverrideCursor(Qt.WaitCursor)  # 鼠标转圈
        try:
            self.progress_bar.setValue(0)  # 初始化进度条
            # 更新进度条提示文本
            self.lbl_progress_info.clear()
            self.lbl_progress_info.setText("训练集特征提取中")
            num = 0  # 已处理的图像数量
            processor = IMGFP.Processor(self.feature_extraction_method)  # 创建图像特征处理器
            self.all_descriptors = {}
            descriptors = []
            self.train_image_paths = []  # 重置路径

            # 重新扫描训练集目录 进度条占30%
            for class_name in os.listdir(self.train_path):  # 扫描训练集下的子目录
                class_dir = os.path.join(self.train_path, class_name)
                if not os.path.isdir(class_dir):
                    continue
                for img_file in os.listdir(class_dir):  # 扫描子目录下的图片文件
                    img_path = os.path.join(class_dir, img_file)
                    # 提取图像特征 (SIFT/ORB)
                    _, desc = processor.extract_features(img_path)
                    if desc is not None:
                        self.all_descriptors[img_path] = desc  # 保存每个训练图像的特征(通过文件路径实现映射)
                        descriptors.append(desc)
                        self.train_image_paths.append((img_path, class_name))
                    # 更新进度条
                    num = num + 1
                    self.progress_bar.setValue(int(num * 30 / self.total_images))
            num = 0  # 进度条分子重置

            # 生成码本 进度条占30%
            self.lbl_progress_info.clear()  # 更新进度条提示文本
            self.lbl_progress_info.setText("训练集码本生成中")
            if len(descriptors) == 0:
                QMessageBox.critical(self, "错误", "无法提取任何训练特征！")
                return

            self.train_codebook = IMGFP.create_codebook(
                descriptors,
                self.codebook_generate_method,
                self.codebook_count,
            )
            self.progress_bar.setValue(40)

            if self.enable_tf_idf:
                # 计算IDF在生成码本后统计每个视觉单词的文档频率
                # 统计每个视觉单词（聚类中心）出现的文档（图像）数量
                num = 0  # 进度条分子重置
                self.lbl_progress_info.setText("IDF计算")
                self.idf = np.zeros(self.codebook_count)  # 用于存储每个视觉单词的IDF值
                # 统计每个视觉单词出现的文档频率（即有多少图像包含该视觉单词）
                for path, _ in self.train_image_paths:
                    _, desc = processor.extract_features(path)
                    if desc is None:
                        continue
                    # 量化到最近的视觉单词
                    visual_words = IMGFP.quantize_features(desc, self.train_codebook)
                    unique_words = np.unique(visual_words)
                    self.idf[unique_words] += 1  # 统计文档频率
                    self.progress_bar.setValue(int(40 + num * 20 / len(self.train_image_paths)))  # 更新进度条
                    num += 1

                # 计算IDF: log(N / df_j)，其中N是总图像数，df_j是包含视觉单词j的图像数量 分母+ 1e-6避免除零
                N = len(self.train_image_paths)
                self.idf = np.log(N / (self.idf + 1e-6))  # +1e-6防止df_j=0
            else:
                self.idf = None
            self.progress_bar.setValue(60)

            # 对每张训练图像进行编码
            self.lbl_progress_info.clear()  # 更新进度条提示文本
            self.lbl_progress_info.setText("训练集特征编码中")
            self.train_encodings = []
            num = 0  # 进度条分子重置
            for path, label in self.train_image_paths:
                _, desc = processor.extract_features(path)
                encoding = IMGFP.encode_features(
                    desc,
                    self.train_codebook,
                    self.feature_encoding_method,
                    idf=self.idf if self.enable_tf_idf else None,
                    enable_tf_idf=self.enable_tf_idf
                )
                self.train_encodings.append(encoding)
                num = num + 1
                self.progress_bar.setValue(int(60 + num * 40 / self.total_images))

            # 过滤无效编码
            valid_encodings = [e for e in self.train_encodings if e is not None]
            if len(valid_encodings) == 0:
                QMessageBox.critical(self, "错误", "所有训练图像编码失败！")
                return

            self.train_encodings = np.array(valid_encodings)
            QMessageBox.information(self, "成功", f"训练完成！码本尺寸：{len(self.train_codebook)}")

            # 自动保存训练数据
            try:
                import pickle
                filename = self.get_training_filename()
                file_path = os.path.join(os.getcwd(), filename)

                training_data = {
                    'train_path': self.train_path,
                    'class_labels': self.class_labels,
                    'class_distribution': self.class_distribution,
                    'train_image_paths': self.train_image_paths,
                    'train_codebook': self.train_codebook,
                    'train_encodings': self.train_encodings,
                    'idf': self.idf,
                    'all_descriptors': self.all_descriptors,
                    'feature_extraction_method': self.feature_extraction_method,
                    'codebook_generate_method': self.codebook_generate_method,
                    'feature_encoding_method': self.feature_encoding_method,
                    'codebook_count': self.codebook_count,
                    'enable_tf_idf': self.enable_tf_idf
                }

                with open(file_path, 'wb') as f:
                    pickle.dump(training_data, f)

                self.lbl_progress_info.setText(f"训练数据已自动保存: {filename}")

            except Exception as e:
                QMessageBox.warning(self, "警告", f"自动保存失败: {str(e)}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"训练失败：{str(e)}")
        finally:
            # QApplication.restoreOverrideCursor()
            self.progress_bar.setValue(0)
            self.lbl_progress_info.clear()
            self.lbl_progress_info.setText("当前处理步骤信息")

    def update_train_dir_info(self, info):
        """更新训练集信息"""
        old_info = self.lbl_train_dir_info.text()
        new_info = old_info + '\n' + info
        self.lbl_train_dir_info.setText(new_info)

    def get_training_filename(self):
        """生成包含算法信息的训练数据文件名"""
        fe_method = {IMGFP.SIFT_pxx: 'SIFT', IMGFP.ORB_pxx: 'ORB'}.get(self.feature_extraction_method, 'DEFAULT')
        cb_method = {IMGFP.KMEANS_pxx: 'KMEANS', IMGFP.VQ_pxx: 'VQ', IMGFP.GMM_pxx: 'GMM'}.get(
            self.codebook_generate_method, 'DEFAULT')
        enc_method = {IMGFP.VLAD_pxx: 'VLAD', IMGFP.BOF_pxx: 'BOF', IMGFP.FV_pxx: 'FV'}.get(
            self.feature_encoding_method, 'DEFAULT')
        tfidf_flag = 'TFIDF' if self.enable_tf_idf else 'noTFIDF'

        return f"training_{fe_method}_{cb_method}_{enc_method}_{self.codebook_count}_{tfidf_flag}.pkl"

    def save_training_data(self):
        """保存训练数据到文件"""
        if not self.train_path or self.train_codebook is None:
            QMessageBox.warning(self, "错误", "没有训练数据可保存")
            return

        # 生成文件名
        filename = self.get_training_filename()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存训练数据", filename, "Pickle Files (*.pkl)"
        )

        if not file_path:
            return

        # 准备保存的数据
        training_data = {
            'train_path': self.train_path,
            'class_labels': self.class_labels,
            'class_distribution': self.class_distribution,
            'train_image_paths': self.train_image_paths,
            'train_codebook': self.train_codebook,
            'train_encodings': self.train_encodings,
            'idf': self.idf,
            'all_descriptors': self.all_descriptors,
            'feature_extraction_method': self.feature_extraction_method,
            'codebook_generate_method': self.codebook_generate_method,
            'feature_encoding_method': self.feature_encoding_method,
            'codebook_count': self.codebook_count,
            'enable_tf_idf': self.enable_tf_idf
        }

        # 保存数据
        try:
            import pickle
            with open(file_path, 'wb') as f:
                pickle.dump(training_data, f)
            QMessageBox.information(self, "成功", f"训练数据已保存到:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")

    def load_training_data(self):
        """从文件加载训练数据"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "加载训练数据", "", "Pickle Files (*.pkl)"
        )

        if not file_path:
            return

        try:
            import pickle
            with open(file_path, 'rb') as f:
                training_data = pickle.load(f)

            # 恢复训练数据
            self.train_path = training_data['train_path']
            self.class_labels = training_data['class_labels']
            self.class_distribution = training_data['class_distribution']
            self.train_image_paths = training_data['train_image_paths']
            self.train_codebook = training_data['train_codebook']
            self.train_encodings = training_data['train_encodings']
            self.idf = training_data['idf']
            self.all_descriptors = training_data['all_descriptors']
            self.feature_extraction_method = training_data['feature_extraction_method']
            self.codebook_generate_method = training_data['codebook_generate_method']
            self.feature_encoding_method = training_data['feature_encoding_method']
            self.codebook_count = training_data['codebook_count']
            self.enable_tf_idf = training_data['enable_tf_idf']

            # 更新UI
            total_images = sum(self.class_distribution.values())
            self._update_train_ui(total_images)  # 更新训练集目录信息

            # 更新算法选择控件的当前索引
            self.cmb_feature_extractor.setCurrentIndex(self.feature_extraction_method)
            self.cmb_codebook_generate.setCurrentIndex(self.codebook_generate_method)
            self.cmb_encoding_method.setCurrentIndex(self.feature_encoding_method)
            self.spb_codebook_size.setValue(self.codebook_count)
            self.rbtn_tfidf.setChecked(self.enable_tf_idf)

            # 启用检测按钮
            self.btn_detect_image.setEnabled(True)

            QMessageBox.information(self, "成功", "训练数据加载成功！")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载失败: {str(e)}")

    def clear_result(self):
        """清除检索结果数据"""
        # 各个指标
        self.elapsed = 0
        self.precision = 0
        self.recall = 0
        # 重置各指标值
        self.lbl_detect_time.setText(f"耗 时：")
        self.lbl_accuracy.setText(f"精 度：")
        self.lbl_recall.setText(f"召回率：")
        self.lbl_ap.setText(f"A P值:")
        self.lbl_map.setText(f"MAP值:")
        self.lbl_result_num.setText("结果图像数:")
        self.lbl_same_class_num.setText("同类图像数:")
        self.lbl_class_num.setText("训练集中该类图像总数:")
        self.lbl_pr_all.setText(f"整体Pr曲线 累计次数: 0")
        self.ap_history.clear()
        self.map_value = 0
        # 清空PR图
        if self.widget_pr_chart_1.layout() is not None:
            while self.widget_pr_chart_1.layout().count():
                child = self.widget_pr_chart_1.layout().takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
        else:
            self.widget_pr_chart_1.setLayout(QtWidgets.QVBoxLayout())

        if self.widget_pr_chart_all.layout() is not None:
            while self.widget_pr_chart_all.layout().count():
                child = self.widget_pr_chart_all.layout().takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
        else:
            self.widget_pr_chart_all.setLayout(QtWidgets.QVBoxLayout())

        # 清空结果图像
        self.list_results.clear()

    def update_feature_extractor(self, index):
        """更新特征提取算法"""
        # if index == IMGFP.ORB_pxx:
        #     QMessageBox.warning(self, "错误", "请勿使用ORB")
        #     self.cmb_feature_extractor.setCurrentIndex(self.feature_extraction_method)
        #     return

        self.feature_extraction_method = index
        print(f"当前特征提取算法: {self.cmb_feature_extractor.itemText(index)}")

    def update_encoding_method(self, index):
        """更新编码方法"""
        if (self.codebook_generate_method == IMGFP.GMM_pxx and
                self.feature_encoding_method == IMGFP.FV_pxx and
                index != IMGFP.FV_pxx):
            # 如果特征编码算法由FV改为别的, 那么码本生成算法不能为GMM
            self.codebook_generate_method = IMGFP.DEFAULT_pxx
            self.cmb_codebook_generate.setCurrentIndex(IMGFP.DEFAULT_pxx)  # 更改UI显示

        self.feature_encoding_method = index
        if index == IMGFP.FV_pxx:  # 当特征编码方法是FV时，码本生成方法应为GMM
            self.codebook_generate_method = IMGFP.GMM_pxx
            self.cmb_codebook_generate.setCurrentIndex(IMGFP.GMM_pxx)  # 更改UI显示

        print(f"当前编码方法: {self.cmb_encoding_method.itemText(index)}")

    def update_codebook_generate(self, index):
        """更新码本生成算法"""
        if (self.codebook_generate_method == IMGFP.GMM_pxx and
                self.feature_encoding_method == IMGFP.FV_pxx and
                index != IMGFP.GMM_pxx):
            # 如果码本生成算法由GMM改为别的, 那么也要更改特征编码为非FV
            self.feature_encoding_method = IMGFP.DEFAULT_pxx
            self.cmb_encoding_method.setCurrentIndex(IMGFP.DEFAULT_pxx)  # 更改UI显示

        self.codebook_generate_method = index
        if index == IMGFP.GMM_pxx:  # 当码本生成方法是GMM时，特征编码方法应为FV
            self.feature_encoding_method = IMGFP.FV_pxx
            self.cmb_encoding_method.setCurrentIndex(IMGFP.FV_pxx)  # 更改UI显示

        print(f"当前码本生成方法: {self.cmb_codebook_generate.itemText(index)}")

    def update_enbal_if_idf(self, index):
        """更新'是否启用IF-IDF'算法"""
        self.enable_tf_idf = index
        print("IF-IDF: " + "启用" if self.enable_tf_idf == 1 else "关闭")

    def update_enbal_qe(self, index):
        """更新'是否启用扩展查询QE'"""
        self.enable_qe = index
        print("扩展查询QE: " + "启用" if self.enable_qe == 1 else "关闭")

    def update_enable_re_sort(self, index):
        """更新重排序算法"""
        self.re_sort_method = index
        if self.re_sort_method == 0:
            print("关闭重排序重排序")
        elif self.re_sort_method == IMGFP.LC_pxx:
            print("启用基于线性组合的重排序算法")
        elif self.re_sort_method == IMGFP.GEOMETRIC_pxx:
            print("启用基于聚类的重排序算法")


if __name__ == '__main__':
    app = QtWidgets.QApplication([])  # 必须在任何 Qt 操作前初始化

    window = main_window()
    window.setWindowTitle(" 图像检索系统 v2.0")  # 设置窗口标题

    # 显示窗口
    window.show()
    # 进入程序的主循环，并通过exit函数确保主循环安全结束(该释放资源的一定要释放)
    sys.exit(app.exec_())
