"""Normalize features"""

import numpy as np


def normalize(features):
    """
    Normalize the features using the mean and standard deviation, calculate zscore
    :param features: numpy array is data for analyze
    :return: features_normalized: numpy array zscored ((x - mean) / std)features
    :return: feature_mean: numpy array
    :return: features_deviation: numpy array
    """

    features_normalized = np.copy(features).astype(float)

    # 计算均值
    features_mean = np.mean(features, 0)

    # 计算标准差
    features_deviation = np.std(features, 0)

    # 标准化操作
    if features.shape[0] > 1:
        features_normalized -= features_mean

    # 防止除以0
    features_deviation[features_deviation == 0] = 1
    features_normalized /= features_deviation

    return features_normalized, features_mean, features_deviation
