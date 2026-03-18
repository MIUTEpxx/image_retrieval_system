# -*- coding: utf-8 -*-
import os
import time
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QInputDialog
import sys
import matplotlib

matplotlib.use('Agg')  # 避免 GUI 线程冲突
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 导入 UI 文件 (View)
import feature_encoding_UI

# 导入业务引擎与相关模块 (Model)
from config import FeatureMethod, CodebookMethod, EncodingMethod, ResortMethod
from retrieval_engine import RetrievalEngine
from metrics import evaluate_retrieval, calculate_map


class main_window(QMainWindow, feature_encoding_UI.Ui_window_feature_encoding):
    """
    系统的主控制器 (Controller)
    负责连接视图 (UI) 与模型 (RetrievalEngine)，并处理用户交互事件。
    """

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 实例化后台核心搜索引擎
        self.engine = RetrievalEngine()

        # --- UI 相关的状态变量 ---
        self.test_path = None
        self.current_test_label = ""
        self.test_set_container = {}  # {类别名:[路径列表]}
        self.ap_history = []  # 保存所有查询的 PR 数据记录
        self.k_value = 10  # Top-K 检索数量

        # 初始化 PR 曲线画布变量
        self.pr_canvas_1 = None
        self.pr_canvas_all = None

        self.init_slot_connection()

    def init_slot_connection(self):
        """信号槽绑定：将 UI 操作映射到具体的事件处理函数"""
        self.btn_select_test_image.clicked.connect(self.select_test_image)
        self.btn_select_train_dir.clicked.connect(self.select_train_dir)
        self.btn_detect_image.clicked.connect(self.start_detect)
        self.btn_clear_result.clicked.connect(self.clear_result)
        self.btn_save_training.clicked.connect(self.save_training_data)
        self.btn_load_training.clicked.connect(self.load_training_data)

        # UI 配置项变动同步给底层引擎 (使用 Lambda 表达式简化代码)
        self.cmb_feature_extractor.currentIndexChanged.connect(
            lambda idx: setattr(self.engine, 'feature_method', FeatureMethod(idx)))
        self.cmb_encoding_method.currentIndexChanged.connect(
            lambda idx: setattr(self.engine, 'encoding_method', EncodingMethod(idx)))
        self.cmb_codebook_generate.currentIndexChanged.connect(
            lambda idx: setattr(self.engine, 'codebook_method', CodebookMethod(idx)))
        self.cmb_re_sort.currentIndexChanged.connect(
            lambda idx: setattr(self.engine, 'resort_method', ResortMethod(idx)))

        self.rbtn_tfidf.toggled.connect(lambda v: setattr(self.engine, 'enable_tfidf', v))
        self.rbtn_qe.toggled.connect(lambda v: setattr(self.engine, 'enable_qe', v))
        self.spb_codebook_size.valueChanged.connect(lambda v: setattr(self.engine, 'codebook_count', v))
        self.spb_knn_k.valueChanged.connect(lambda v: setattr(self, 'k_value', v))

    # =========== 1. 文件选择与 UI 更新模块 (View Helpers) ============

    def select_test_image(self):
        """弹出对话框选择测试图片或目录"""
        if not self.engine.class_labels:
            QMessageBox.warning(self, "注意", "请先选择训练集以确定类别标签")
            return

        if self.cmb_test_format.currentIndex() == 0:  # 单张模式
            path, _ = QFileDialog.getOpenFileName(self, "选择测试图片", "", "Images (*.jpg *.png *.bmp)")
            if path:
                self.test_path = path
                self.lbl_test_image_path.setText(path)
                # 自动推断类别：假设测试图在类名命名的文件夹里
                parent_dir = os.path.basename(os.path.dirname(path))
                if parent_dir in self.engine.class_labels:
                    self.current_test_label = parent_dir
                else:
                    label, ok = QInputDialog.getItem(self, "选择类别", "请指定测试图的真实类别：",
                                                     self.engine.class_labels, 0, False)
                    if ok: self.current_test_label = label

                self.label_test_img.setText(f"测试图像 (类别: {self.current_test_label})")
                self.show_test_image(path)
        else:  # 目录模式
            test_dir = QFileDialog.getExistingDirectory(self, "选择测试集目录")
            if test_dir:
                self.test_path = test_dir
                self.lbl_test_image_path.setText(test_dir)
                # 扫描目录下的子文件夹作为类别
                self.test_set_container = {}
                for cls in os.listdir(test_dir):
                    cls_path = os.path.join(test_dir, cls)
                    if os.path.isdir(cls_path):
                        imgs = [os.path.join(cls_path, f) for f in os.listdir(cls_path) if
                                f.lower().endswith(('.jpg', '.png'))]
                        if imgs: self.test_set_container[cls] = imgs

    def show_test_image(self, path):
        """在界面显示预览图"""
        pixmap = QPixmap(path)
        scaled_pixmap = pixmap.scaled(self.lbl_test_image.width(), self.lbl_test_image.height(), Qt.KeepAspectRatio,
                                      Qt.SmoothTransformation)
        self.lbl_test_image.setPixmap(scaled_pixmap)

    def select_train_dir(self):
        """选择训练集并自动触发训练流程"""
        path = QFileDialog.getExistingDirectory(self, "选择训练集目录")
        if not path: return

        # UI 界面提示更新
        self.lbl_train_dir_path.setText(f"训练集路径: {path}")
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            # 自动开始底层训练流水线，传入 UI 更新回调函数
            self.engine.train(path, progress_callback=self.update_progress)

            # 训练完成后更新 UI 统计信息
            info_lines = [
                f"总样本数: {len(self.engine.train_image_paths)}",
                f"类别数: {len(self.engine.class_labels)}",
                "类别分布:"
            ]
            for cls, count in self.engine.class_distribution.items():
                info_lines.append(f"   • {cls}: {count}张")
            self.lbl_train_dir_info.setText("\n".join(info_lines))

            self.btn_detect_image.setEnabled(True)
            QMessageBox.information(self, "成功", "训练完成！可以开始检测。")

        except Exception as e:
            QMessageBox.critical(self, "训练失败", f"异常: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()
            self.update_progress(0, "当前处理步骤信息")

    # =========== 2. 核心检测业务流 (Business Logic Call) ============

    def start_detect(self):
        """触发检测流程"""
        if not self.test_path or self.engine.train_encodings is None:
            QMessageBox.warning(self, "错误", "请先准备好测试图并加载训练模型！")
            return

        if self.cmb_test_format.currentIndex() == 0:
            self._process_single_query(self.test_path, self.current_test_label)
        else:
            self._process_batch_query()

    def _process_single_query(self, query_path, true_label):
        """单张图像检索流程"""
        start_time = time.time()
        try:
            # 核心业务调用引擎
            results = self.engine.search(query_path, top_k=self.k_value)
            elapsed = time.time() - start_time

            # 计算评价指标
            total_relevant = self.engine.class_distribution.get(true_label, 0)
            metrics = evaluate_retrieval(results, true_label, total_relevant, self.k_value)

            self.ap_history.append(metrics)
            mAP = calculate_map(self.ap_history)

            # UI 渲染
            self.show_result(results)
            self._update_metrics_ui(elapsed, metrics, mAP, total_relevant)
            self.plot_pr_curve(metrics['recall_points'], metrics['precision_points'], metrics['ap'],
                               self.widget_pr_chart_1)
            self.update_overall_pr_curve()

        except Exception as e:
            QMessageBox.warning(self, "检索失败", f"异常: {str(e)}")

    def _process_batch_query(self):
        """批量检索"""
        total = 0
        for class_name, img_paths in self.test_set_container.items():
            for img_path in img_paths:
                self._process_single_query(img_path, class_name)
                total += 1
        QMessageBox.information(self, "完成", f"批量处理完成，共 {total} 张图。")

    # =========== 3. 结果展示与绘图模块 (Rendering) ============

    def show_result(self, results):
        """在 QListWidget 中展示结果缩略图"""
        self.list_results.clear()
        item_size = QtCore.QSize(200, 180)
        self.list_results.setGridSize(item_size)

        for res in results:
            item = QtWidgets.QListWidgetItem()
            widget = self._create_result_item(res['label'], res['path'], res.get('dist', 0))
            item.setSizeHint(item_size)
            self.list_results.addItem(item)
            self.list_results.setItemWidget(item, widget)

    def _create_result_item(self, label, path, dist):
        """创建单个结果卡片"""
        container = QtWidgets.QGroupBox()
        layout = QtWidgets.QVBoxLayout(container)

        img_label = QtWidgets.QLabel()
        pix = QPixmap(path).scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        img_label.setPixmap(pix)
        img_label.setAlignment(Qt.AlignCenter)

        info = QtWidgets.QLabel(f"<b>{label}</b><br>Dist: {dist:.3f}")
        info.setAlignment(Qt.AlignCenter)

        layout.addWidget(img_label)
        layout.addWidget(info)
        container.setStyleSheet("QGroupBox { border: 1px solid silver; border-radius: 5px; }")
        return container

    def plot_pr_curve(self, recall, precision, ap, target_widget):
        """使用 Matplotlib 在 Qt 控件中画 PR 曲线"""
        if target_widget.layout() is None:
            target_widget.setLayout(QtWidgets.QVBoxLayout())

        # 清理旧图
        while target_widget.layout().count():
            child = target_widget.layout().takeAt(0)
            if child.widget(): child.widget().deleteLater()

        fig = Figure(figsize=(4, 3))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.plot(recall, precision, 'r-', label=f'AP={ap:.3f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('P-R Curve')
        ax.legend()
        ax.grid(True)

        target_widget.layout().addWidget(canvas)

    def update_overall_pr_curve(self):
        """计算并更新累积的 mAP 曲线 """
        if not self.ap_history: return

        # 确定所有记录的最小召回范围
        min_recalls = [np.min(record['recall_points']) for record in self.ap_history if len(record['recall_points']) > 0]
        if not min_recalls: return

        # 从最大最小召回值开始插值
        max_min_recall = max(min_recalls)
        recall_levels = np.linspace(max_min_recall, 1, 100)
        interp_precisions =[]

        for record in self.ap_history:
            recalls = record['recall_points']
            precisions = record['precision_points']
            if len(recalls) == 0: continue

            # 确保单调递增
            sorted_indices = np.argsort(recalls)
            recalls_sorted = recalls[sorted_indices]
            precisions_sorted = precisions[sorted_indices]

            min_precision = precisions_sorted[0]
            max_recall = recalls_sorted[-1]

            # 线性插值
            interp_prec = np.interp(
                recall_levels, recalls_sorted, precisions_sorted,
                left=min_precision,
                right=precisions_sorted[-1] if len(precisions_sorted) > 0 else min_precision
            )
            interp_prec[recall_levels > max_recall] = precisions_sorted[-1] if len(precisions_sorted) > 0 else min_precision
            interp_precisions.append(interp_prec)

        valid_precisions = [p for p in interp_precisions if np.any(p > 0)]
        if not valid_precisions: return

        avg_precision = np.mean(valid_precisions, axis=0)
        mAP = calculate_map(self.ap_history)

        first_valid = np.argmax(avg_precision > 0)
        valid_recalls = recall_levels[first_valid:]
        valid_precision = avg_precision[first_valid:]

        self.plot_pr_curve(valid_recalls, valid_precision, mAP, self.widget_pr_chart_all)

    def _update_metrics_ui(self, elapsed, metrics, mAP, total_relevant):
        """更新指标面板文本 (修复显示内容)"""
        self.lbl_detect_time.setText(f"耗时：{elapsed:.3f}秒")
        self.lbl_accuracy.setText(f"精度：{metrics['precision']:.2%}")
        self.lbl_recall.setText(f"召回率：{metrics['recall']:.2%}")
        self.lbl_ap.setText(f"A P值: {metrics['ap']:.3f}")
        self.lbl_map.setText(f"MAP值: {mAP:.3f}")
        self.lbl_result_num.setText(f"结果图像数: {self.k_value}")        # 修复显示项
        self.lbl_same_class_num.setText(f"同类图像数: {metrics['correct_count']}") # 修复显示项
        self.lbl_class_num.setText(f"训练集中该类图像总数: {total_relevant}")
        self.lbl_pr_all.setText(f"整体Pr曲线 累计次数: {len(self.ap_history)}")

    def clear_result(self):
        """清空数据重置界面"""
        self.list_results.clear()
        self.ap_history = []

        # 还原文字标签状态
        self.lbl_detect_time.setText("耗时：")
        self.lbl_accuracy.setText("精度：")
        self.lbl_recall.setText("召回率：")
        self.lbl_ap.setText("A P值：")
        self.lbl_map.setText("MAP值：")
        self.lbl_result_num.setText("结果图像数:")
        self.lbl_same_class_num.setText("同类图像数:")
        self.lbl_class_num.setText("训练集中该类图像总数:")
        self.lbl_pr_all.setText("整体Pr曲线 累计次数: 0")

        # 彻底清空 Matplotlib 画布，防止残影
        for widget in [self.widget_pr_chart_1, self.widget_pr_chart_all]:
            if widget.layout() is not None:
                while widget.layout().count():
                    child = widget.layout().takeAt(0)
                    if child.widget(): child.widget().deleteLater()

    def save_training_data(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存模型", "model.pkl", "Pickle (*.pkl)")
        if path: self.engine.save_model(path)

    def load_training_data(self):
        """加载模型并强制同步 UI 控制面板"""
        path, _ = QFileDialog.getOpenFileName(self, "加载模型", "", "Pickle (*.pkl)")
        if not path: return
        try:
            self.engine.load_model(path)

            # 【修复核心】同步引擎参数到 UI 界面，防止后端跑 ORB，前端显示 SIFT
            self.cmb_feature_extractor.setCurrentIndex(self.engine.feature_method.value)
            self.cmb_codebook_generate.setCurrentIndex(self.engine.codebook_method.value)
            self.cmb_encoding_method.setCurrentIndex(self.engine.encoding_method.value)
            self.spb_codebook_size.setValue(self.engine.codebook_count)
            self.rbtn_tfidf.setChecked(self.engine.enable_tfidf)

            # 更新训练集信息面板
            total = len(self.engine.train_image_paths)
            self.lbl_train_dir_path.setText(f"模型已加载: {os.path.basename(path)}")
            self.lbl_train_dir_info.setText(f"总样本数: {total}\n类别数: {len(self.engine.class_labels)}")
            self.btn_detect_image.setEnabled(True)
            QMessageBox.information(self, "成功", "模型加载且 UI 同步成功！")
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"模型版本不兼容或损坏: {str(e)}")

        # =========== 进度条更新回调 (重要：连接 Engine 与 UI) ============
    def update_progress(self, value: int, msg: str):
        """
        UI 回调函数：用于接收底层引擎抛出的训练进度并更新界面
        :param value: 进度值 (0-100)
        :param msg: 当前处理的步骤描述文本
        """
        # 1. 更新进度条数值
        self.progress_bar.setValue(value)

        # 2. 更新进度条旁边的标签文字
        self.lbl_progress_info.setText(msg)

        # 3. 强制刷新 UI 事件循环
        # 知识点：在执行耗时的特征提取循环时，如果不调用 processEvents()，
        # UI 线程会被阻塞，导致界面看起来“卡死”或“无响应”。
        # 这一行代码告诉 Qt：先停下手中的重活，去把界面画一下，然后再继续。
        QApplication.processEvents()

# ==========================================
# 程序入口
# ==========================================
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = main_window()
    window.setWindowTitle("高级图像检索系统 (重构版) - 杭电复试演示版")
    window.show()
    sys.exit(app.exec_())