import numpy as np


def generate_sinusoids(dataset, sinusoid_degree):
    """
    sin(x). 正弦波，从中让机器学出周期性特征
    :param dataset: 特征矩阵
    :param sinusoid_degree: 计算正弦的阶，sin(x), sin(2x)..., sin(degree)
    :return: numpy array [(dataset.shape[0], sinusoid_degree * dataset.shape[1])]
    """

    num_examples = dataset.shape[0]
    sinusoids = np.empty((num_examples, 0))

    for degree in range(1, sinusoid_degree + 1):
        sinusoid_features = np.sin(degree * dataset)
        sinusoids = np.concatenate((sinusoids, sinusoid_features), axis=1)
        
    return sinusoids
