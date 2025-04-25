import os
import time

import numpy as np
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

        # 算法选择
        self.cmb_feature_extractor.currentIndexChanged.connect(self.update_feature_extractor)  # 特征提取算法
        self.cmb_encoding_method.currentIndexChanged.connect(self.update_encoding_method)  # 特征编码算法
        self.cmb_codebook_generate.currentIndexChanged.connect(self.update_codebook_generate)  # 码本生成算法
        self.rbtn_tfidf.toggled.connect(self.update_enbal_if_idf)
        # 参数输入
        self.spb_codebook_size.valueChanged.connect(lambda v: setattr(self, 'codebook_count', v))
        self.spb_knn_k.valueChanged.connect(lambda v: setattr(self, 'k_value', v))
        self.dspb_match_threshold.valueChanged.connect(lambda v: setattr(self, 'feature_matching_threshold', v))

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
        """测试图特征提取与编码"""
        processor = IMGFP.Processor(self.feature_extraction_method)
        _, desc = processor.extract_features(self.test_path)
        if desc is None:
            QMessageBox.warning(self, "错误", "无法提取测试图像特征")
            return

        # 生成测试编码
        test_encoding = IMGFP.encode_features(
            desc,
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

        #  获取Top-K结果
        k = min(self.k_value, len(distances))
        nearest_indices = np.argpartition(distances, k)[:k]
        sorted_indices = nearest_indices[np.argsort(distances[nearest_indices])]

        # 构建结果列表
        results = []  # 格式 [(绝对路径,类别,相似度)]
        for idx in sorted_indices:
            img_path, label = self.train_image_paths[idx]
            results.append((label, img_path, distances[idx]))

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

    def show_result(self, results):
        """显示检测结果图像和信息"""
        self.list_results.clear()

        item_size = QtCore.QSize(200, 150)  # 统一项目尺寸
        self.list_results.setGridSize(item_size)

        for label, path, dist in results:
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

            # 插值时使用当前记录首点精度作为左边界
            interp_prec = np.interp(
                recall_levels,
                recalls_sorted,
                precisions_sorted,
                left=min_precision,  # 使用当前记录的最小召回点精度
                right=0.0
            )
            interp_precisions.append(interp_prec)

        # 计算平均精度时忽略全零数据
        valid_precisions = [p for p in interp_precisions if np.any(p > 0)]
        if not valid_precisions:
            return

        avg_precision = np.mean(valid_precisions, axis=0)
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
        """选择测试图片"""
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
            # self.lbl_detect_time.setText(f"耗时：")
            # self.lbl_accuracy.setText(f"精度：")
            # self.lbl_recall.setText(f"召回率：")
            # self.lbl_ap.setText(f"A P值:")
            # self.lbl_map.setText(f"MAP值:")
            # self.lbl_result_num.setText("结果图像数:")
            # self.lbl_same_class_num.setText("同类图像数:")
            # # if self.current_test_label is not None and self.class_distribution[]:
            # self.lbl_class_num.setText("训练集中该类图像总数:")
            # self.ap_history.clear()
            # self.map_value = 0

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
            all_descriptors = []
            self.train_image_paths = []  # 重置路径

            # 重新扫描训练集目录 进度条占30%
            for class_name in os.listdir(self.train_path):  # 扫描训练集下的子目录
                class_dir = os.path.join(self.train_path, class_name)
                if not os.path.isdir(class_dir):
                    continue
                for img_file in os.listdir(class_dir):  # 扫描子目录下的图片文件
                    img_path = os.path.join(class_dir, img_file)
                    # 提取图像特征 (SIFT)
                    _, desc = processor.extract_features(img_path)
                    if desc is not None:
                        all_descriptors.append(desc)
                        self.train_image_paths.append((img_path, class_name))
                    # 更新进度条  待实现
                    num = num + 1
                    self.progress_bar.setValue(int(num * 30 / self.total_images))
            num = 0  # 进度条分子重置

            # 生成码本 进度条占30%
            self.lbl_progress_info.clear()  # 更新进度条提示文本
            self.lbl_progress_info.setText("训练集码本生成中")
            if len(all_descriptors) == 0:
                QMessageBox.critical(self, "错误", "无法提取任何训练特征！")
                return

            self.train_codebook = IMGFP.create_codebook(
                all_descriptors,
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

        print(f"当前码本生成方法: {self.cmb_encoding_method.itemText(index)}")

    def update_enbal_if_idf(self, index):
        """更新'是否启用IF-IDF'算法"""
        self.enable_tf_idf = index
        print(self.enable_tf_idf)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])  # 必须在任何 Qt 操作前初始化

    window = main_window()
    window.setWindowTitle(" 图像检索系统 v2.0")  # 设置窗口标题

    # 显示窗口
    window.show()
    # 进入程序的主循环，并通过exit函数确保主循环安全结束(该释放资源的一定要释放)
    sys.exit(app.exec_())
