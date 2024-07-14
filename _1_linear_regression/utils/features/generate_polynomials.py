"""Add polynomial features to the features set"""

import numpy as np
from .normalize import normalize


def generate_polynomials(dataset, polynomial_degree, normalize_data=False):
    """变换方法：
    x1, x2, x1^2, x2^2, x1*x2, x1*x2^2, etc.
    上述多项式可以理解为从两个变量，生成的各种幂次和组合。这些多项式特征可以用于机器学习模型，以捕捉输入变量之间的非线性关系
    """

    features_split = np.array_split(dataset, 2, axis=1)
    dataset_1 = features_split[0]
    dataset_2 = features_split[1]

    (num_examples_1, num_features_1) = dataset_1.shape
    (num_examples_2, num_features_2) = dataset_2.shape

    if num_examples_1 != num_examples_2:
        raise ValueError('Can not generate polynomials for two sets with different number of rows')

    if num_features_1 == 0 and num_features_2 == 0:
        raise ValueError('Can not generate polynomials for two sets with no columns')

    if num_features_1 == 0:
        dataset_1 = dataset_2
    elif num_features_2 == 0:
        dataset_2 = dataset_1

    num_features = num_features_1 if num_features_1 < num_examples_2 else num_features_2
    dataset_1 = dataset_1[:, :num_features]
    dataset_2 = dataset_2[:, :num_features]

    polynomials = np.empty((num_examples_1, 0))

    # 1 ~ polynomial_degree
    # (1,0),(1,1),(2,0),(2,1),(2,2),(3,0),(3,1),(3,2),(3,3)
    for i in range(1, polynomial_degree + 1):
        # 0 ~ i
        for j in range(i + 1):
            # **是^的效果，表示幂次
            # x1, x2, x1^2, x1*x2, x2^2, x1^3, x1^2*x2, x1*x2^2, x2^3
            polynomial_feature = (dataset_1 ** (i - j)) * (dataset_2 ** j)
            polynomials = np.concatenate((polynomials, polynomial_feature), axis=1)

    if normalize_data:
        polynomials = normalize(polynomials)[0]

    return polynomials
