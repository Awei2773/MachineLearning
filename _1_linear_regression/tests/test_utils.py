import numpy as np

import _1_linear_regression.utils.features.normalize as normalize
import _1_linear_regression.utils.features.generate_sinusoids as generate_sinusoids

def test_normalize():
    features = np.array([[1, 2, 3, 4, 5],
             [2, 3, 3, 5, 6]])

    (features_normalized, features_mean, features_deviation) = normalize(features)

    assert np.array_equal(np.array([1.5, 2.5, 3.0, 4.5, 5.5]), features_mean)
    assert np.array_equal(np.array([0.5, 0.5, 1.0, 0.5, 0.5]), features_deviation)
    assert np.array_equal(np.array([[-1.0, -1.0,  0.0, -1.0, -1.0],
                                    [ 1.0,  1.0,  0.0,  1.0,  1.0]]), features_normalized)

def test_generate_sinusoids():
    dataset = np.array([[1, 2],
                        [3, 4],
                        [5, 6]])
    sinusoid_degree = 3
    sinusoids = generate_sinusoids(dataset, sinusoid_degree)

    assert sinusoids.shape[0] == 3
    assert sinusoids.shape[1] == 6
