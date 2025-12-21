


def x_squared(x):
    return x**2



def x_cuded(x):
    return x**3



def x_cross_multiply(x):
    """
    Given input x of shape (n_samples, n_features),
    return the cross-multiplied features of shape (n_samples, n_cross_features),
    where n_cross_features = n_features * (n_features - 1) / 2
    """
    import numpy as np
    n_samples, n_features = x.shape
    cross_features = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            cross_features.append((x[:, i] * x[:, j]).reshape(-1, 1))
    return np.hstack(cross_features)

if __name__ == "__main__":
    # Test the feature augmentation functions
    import numpy as np

    x = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
    print("x:", x)

    x2 = x_squared(x)
    print("x squared:", x2)

    x3 = x_cuded(x)
    print("x cubed:", x3)

    x_cross = x_cross_multiply(x)
    print("x cross-multiplied:", x_cross)