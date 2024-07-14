"""Prepares the dataset for training"""

import numpy as np
from .normalize import normalize
from .generate_sinusoids import generate_sinusoids
from .generate_polynomials import generate_polynomials


def prepare_for_training(data, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
    """"
    Prepares the dataset for training and validation
    :param data: 特征矩阵
    :param polynomial_degree: degrees of polynomial
    :param sinusoid_degree: degrees of sinusoid
    :param normalize_data: whether to normalize
    :return: [one, data_normalized, sinusoids, polynomials], features_mean, features_deviation
    """

    # 计算样本总数
    num_examples = data.shape[0]

    data_processed = np.copy(data)

    # 预处理
    features_mean = 0
    features_deviation = 0
    data_normalized = data_processed
    if normalize_data:
        (
            data_normalized,
            features_mean,
            features_deviation
        ) = normalize(data_processed)

        data_processed = data_normalized

    # 特征变换sinusoidal(正弦变换)， axis在numpy库中表示进行运算的轴(0: 行，1：列，多维数组以此类推)
    if sinusoid_degree > 0:
        sinusoids = generate_sinusoids(data_normalized, sinusoid_degree)
        # concatenate, 指定在axis方向上将两个矩阵进行连接
        data_processed = np.concatenate((data_processed, sinusoids), axis=1)

    # 特征变换polynomial(多项式变换)
    if polynomial_degree > 0:
        polynomials = generate_polynomials(data_normalized, polynomial_degree, normalize_data)
        data_processed = np.concatenate((data_processed, polynomials), axis=1)

    # 加一列1，效果和concatenate(array, axis=1)相同，水平方向上进行矩阵拼接
    data_processed = np.hstack((np.ones((num_examples, 1)), data_processed))

    return data_processed, features_mean, features_deviation
