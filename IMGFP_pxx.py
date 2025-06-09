import cv2
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from skimage.feature import hog, local_binary_pattern  # 导入HOG和LBP函数
import numpy as np
import cv2

# 算法选项常量
DEFAULT_pxx = 0
# 特征提取算法
SIFT_pxx = 0
ORB_pxx = 1
# 码本生成算法
KMEANS_pxx = 0
VQ_pxx = 1
GMM_pxx = 2
# 编码算法
VLAD_pxx = 0
BOF_pxx = 1
FV_pxx = 2
# 重排序
LC_pxx = 1  # 线性组合重排序
GEOMETRIC_pxx = 2  # 基于几何验证的重排序


class Processor:
    def __init__(self, method=DEFAULT_pxx):
        self.method = method
        self.feature_extractor = None
        if self.method == ORB_pxx:
            self.feature_extractor = cv2.ORB_create(
                nfeatures=500,  # 控制特征点数量
                scaleFactor=1.2,  # 金字塔尺度因子
                nlevels=8,  # 金字塔层数
                edgeThreshold=15  # 边缘阈值与SIFT一致
            )
        else:  # 默认使用SIFT
            self.feature_extractor = cv2.SIFT_create(
                # contrastThreshold=0.01,
                # edgeThreshold=15
            )

        # 新增全局特征提取参数
        self.color_bins = 64  # 颜色直方图分箱数
        self.hog_params = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2)
        }

    def extract_global_features(self, image_path):
        """提取颜色/形状/纹理特征"""
        # img = cv2.imread(image_path)
        # features = {}
        #
        # # 颜色特征（HSV直方图）
        # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # hist_color = cv2.calcHist([hsv], [0, 1], None, [self.color_bins] * 2, [0, 180, 0, 256])
        # hist_color = cv2.normalize(hist_color, None).flatten()
        # features['color'] = hist_color

        img = cv2.imread(image_path)
        features = {}

        # 颜色特征（三维LAB直方图）
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab = cv2.normalize(lab, None, 0, 255, cv2.NORM_MINMAX)  # 归一化到[0,255]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 计算带空间加权的三维直方图
        hist = cv2.calcHist(
            images=[lab],
            channels=[0, 1, 2],
            mask=None,
            histSize=[8, 8, 8],  # 每个通道8个bin
            ranges=[0, 256, 0, 256, 0, 256]  # 注意每个通道的范围
        )
        hist = cv2.normalize(hist, None).flatten()
        features['color'] = hist

        # # 形状特征（HOG）
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # hog_feat = hog(  # 直接调用hog函数，无需feature.前缀
        #     gray,
        #     orientations=self.hog_params['orientations'],
        #     pixels_per_cell=self.hog_params['pixels_per_cell'],
        #     cells_per_block=self.hog_params['cells_per_block'],
        #     channel_axis=None  # 灰度图无需通道轴
        # )
        # features['shape'] = hog_feat

        self.hog_params = {
            'orientations': 12,  # 增加方向分辨率
            'pixels_per_cell': (16, 16),  # 增大cell尺寸
            'cells_per_block': (3, 3),  # 扩大block范围
            'transform_sqrt': True  # Gamma校正
        }

        # 添加PCA降维

        from sklearn.decomposition import PCA
        hog_feat = hog(  # 直接调用hog函数，无需feature.前缀
            gray,
            orientations=self.hog_params['orientations'],
            pixels_per_cell=self.hog_params['pixels_per_cell'],
            cells_per_block=self.hog_params['cells_per_block'],
            channel_axis=None  # 灰度图无需通道轴
        )

        features['shape'] = hog_feat.astype(np.float32)  # 直接使用原始HOG特征

        # 纹理特征（LBP）
        lbp = local_binary_pattern(  # 直接调用local_binary_pattern
            gray,
            P=24,  # 圆形邻域采样点数
            R=3,  # 邻域半径
            method='uniform'  # 统一模式减少维度
        )
        hist_texture, _ = np.histogram(lbp, bins=256, range=(0, 256))
        hist_texture = hist_texture.astype(np.float32)
        features['texture'] = hist_texture

        # 对HOG和LBP特征进行L2归一化，避免某些特征范围过大
        features['shape'] = hog_feat / np.linalg.norm(hog_feat + 1e-6)
        features['texture'] = hist_texture / np.linalg.norm(hist_texture + 1e-6)
        return features

    def extract_features(self, image_path):
        """提取图像特征点和描述子"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None
        keypoints, descriptors = self.feature_extractor.detectAndCompute(img, None)

        # 处理ORB描述子的数据类型
        if descriptors is not None and self.method == ORB_pxx:
            # 将二进制描述子转换为float32类型
            descriptors = descriptors.astype(np.float32)
        return keypoints, descriptors


def create_codebook(descriptors_list, method, n_clusters):
    """生成码本"""
    # 预处理：过滤无效描述符
    all_descriptors = np.vstack(descriptors_list)
    valid_descs = [d for d in descriptors_list if d is not None and len(d) > 0]
    if not valid_descs:
        raise ValueError("无有效特征描述符可用于生成码本")
    all_descs = np.vstack(valid_descs)

    if method == KMEANS_pxx:
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=1024,
            max_iter=50,
            random_state=0
        )

        # 分批部分拟合 防止卡死
        num = 0  # 已处理的数量
        total_num = len(descriptors_list)  # 总共需要处理的数量
        for desc in descriptors_list:
            if desc is None or len(desc) == 0:
                continue
            kmeans.partial_fit(desc)  # 增量训练

        return kmeans.cluster_centers_
    elif method == GMM_pxx:
        # 合并并过滤描述符
        valid_descs = [d for d in descriptors_list if d is not None and len(d) > 0]
        all_descs = np.vstack(valid_descs)

        # 样本数量检查
        if len(all_descs) < n_clusters * 5:
            raise ValueError(
                f"GMM需要至少 {n_clusters * 5} 个样本，当前只有 {len(all_descs)} 个"
            )

        # 使用MiniBatchKMeans初始化
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024)
        kmeans.fit(all_descs)

        # 训练GMM
        gmm = GaussianMixture(
            n_components=n_clusters,
            means_init=kmeans.cluster_centers_,  # 用kmeans加速初始化
            covariance_type='diag',
            max_iter=100,
            verbose=2
        )
        gmm.fit(all_descs)
        return gmm

    elif method == VQ_pxx:
        # 数据量校验
        min_samples = max(n_clusters * 5, 1000)  # 至少1000样本或5倍聚类数
        if len(all_descs) < min_samples:
            raise ValueError(
                f"VQ需要至少 {min_samples} 个样本，当前只有 {len(all_descs)} 个。\n"
                f"建议：1. 增加训练图片 2. 降低码本尺寸 3. 调整特征提取参数"
            )

        # 标准化数据
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        normalized_descs = scaler.fit_transform(all_descs)

        # 完整版K-means（相比MiniBatch更精确但更慢）

        vq = MiniBatchKMeans(
            n_clusters=n_clusters,
            init='k-means++',
            batch_size=1024,  # 必须指定批大小
            max_iter=100,  # 减少迭代次数
            compute_labels=False,  # 提升大数据的处理速度
            verbose=1
        )

        # 分块训练（防止内存溢出）
        chunk_size = 100000  # 每次处理10万样本
        for i in range(0, len(normalized_descs), chunk_size):
            chunk = all_descs[i:i + chunk_size]
            vq.partial_fit(chunk)  # 统一使用partial_fit

        return vq.cluster_centers_

    else:
        raise ValueError("Unsupported codebook method")


def encode_features(descriptors, codebook, method, enable_tf_idf=False, idf=None):
    """特征编码"""
    if descriptors is None or len(descriptors) == 0:
        return None

    if method == BOF_pxx:
        # BoF 编码
        nn = NearestNeighbors(n_neighbors=1).fit(codebook)
        _, indices = nn.kneighbors(descriptors)
        indices = indices.flatten()  # 转换为一维索引数组
        # 生成原始词频直方图（非归一化）
        hist, _ = np.histogram(indices, bins=range(len(codebook) + 1), density=False)

        # 计算TF（词频归一化到[0, 1]）
        # tf = hist / (hist.sum() + 1e-6)  # 防止除零
        # 应用log(1 + TF)缩放代替归一化
        tf = np.log(1 + hist)  # 降低高频单词影响

        # 是否启用TF-IDF
        if enable_tf_idf and idf is not None:
            hist = tf * idf  # TF-IDF加权
        else:
            hist = tf  # 仅使用TF

        # L2归一化 确保输出向量单位长度，便于距离计算
        hist = hist / (np.linalg.norm(hist) + 1e-6)
        return hist.flatten()

    elif method == VLAD_pxx:
        # VLAD 编码
        nn = NearestNeighbors(n_neighbors=1).fit(codebook)
        _, indices = nn.kneighbors(descriptors)
        # indices = indices.flatten()  # 转换为一维索引数组
        # vlad = np.zeros((len(codebook), descriptors.shape[1]))
        # for i, idx in enumerate(indices):
        #     residual = descriptors[i] - codebook[idx]
        #     vlad[idx] += residual

        # 修改VLAD编码逻辑
        vlad = np.zeros((len(codebook), descriptors.shape[1]))
        indices = quantize_features(descriptors, codebook)
        for k in range(len(codebook)):
            mask = (indices == k)
            if np.any(mask):
                vlad[k] = (descriptors[mask] - codebook[k]).sum(axis=0)

        # 应用IDF加权（仅对残差块加权：对每个聚类中心对应的残差块乘以其IDF值）
        if enable_tf_idf and idf is not None:
            for k in range(len(codebook)):
                vlad[k] *= idf[k]  # 每个聚类中心残差乘对应IDF

        # 展平并L2归一化 避免因IDF加权导致向量尺度变化
        vlad = vlad.flatten()
        norm = np.linalg.norm(vlad)
        if norm > 0:
            vlad /= norm

        return vlad

    elif method == FV_pxx and isinstance(codebook, GaussianMixture):
        # Fisher Vector 编码
        means = codebook.means_
        covs = codebook.covariances_
        weights = codebook.weights_

        # 计算后验概率
        post = codebook.predict_proba(descriptors)

        # 计算梯度
        d_means = []
        d_sigmas = []
        for k in range(codebook.n_components):
            post_k = post[:, k]
            diff = descriptors - means[k]
            inv_sigma = 1 / np.sqrt(covs[k])

            # 关于均值的梯度
            grad_mean = post_k[:, None] * diff * inv_sigma
            d_means.append(grad_mean.sum(axis=0))

            # 关于方差的梯度
            grad_sigma = post_k[:, None] * (diff ** 2 * inv_sigma ** 3 - inv_sigma)
            d_sigmas.append(grad_sigma.sum(axis=0))

        fv = np.concatenate([np.concatenate(d_means), np.concatenate(d_sigmas)])
        return fv / np.linalg.norm(fv)

    else:
        raise ValueError("Unsupported encoding method")


def quantize_features(descriptors, codebook):
    """
    将描述子量化到最近的视觉单词（码本中的聚类中心）
    descriptors: 描述子数组, 形状为 (n_descriptors, n_features)
    codebook: 码本（聚类中心数组）, 形状为 (n_clusters, n_features)
    return: indices: 每个描述子对应的最近视觉单词索引数组, 形状为 (n_descriptors,)
    """
    if descriptors is None or len(descriptors) == 0:
        return None
    if codebook is None or len(codebook) == 0:
        return None

    # 使用最近邻算法（默认欧氏距离）
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(codebook)
    _, indices = nn.kneighbors(descriptors)

    # 转换为一维索引数组
    return indices.flatten().astype(int)
