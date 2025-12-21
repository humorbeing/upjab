from upjab_ActGP.transforms.design_data.feature_augmentation import x_cuded
from upjab_ActGP.transforms.design_data.feature_augmentation import x_squared
from upjab_ActGP.transforms.design_data.feature_augmentation import x_cross_multiply
import numpy as np

def FA01(x_train):
    x_train_square = x_squared(x_train)
    x_train_cube = x_cuded(x_train)

    x_train = np.hstack((x_train, x_train_square, x_train_cube))
    return x_train


def FA02(x_train):
    x_train_square = x_squared(x_train)
    # x_train_cube = x_cuded(x_train)

    x_train = np.hstack((x_train, x_train_square))
    return x_train


def FA03(x_train):
    x_cross = x_cross_multiply(x_train)

    x_train = np.hstack((x_train, x_cross))
    return x_train

from upjab_ActGP.transforms.design_data.feature_augmentation_02 import full_cross_multiply_norepeat

def FA04(x_train, FAR=2):
    x_cross = full_cross_multiply_norepeat(x_train, FAR=FAR)

    # x_train = np.hstack((x_train, x_cross))
    return x_cross


from upjab_ActGP.transforms.design_data.feature_augmentation_02 import full_cross_multiply_full

def FA05(x_train, FAR=2):
    x_cross = full_cross_multiply_full(x_train, FAR=FAR)

    # x_train = np.hstack((x_train, x_cross))
    return x_cross
    

if __name__ == "__main__":
    # Test the feature augmentation functions
    import numpy as np

    x = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
    print("x:", x)

    x_fa01 = FA01(x)
    print("x with FA01:", x_fa01)

    x_fa02 = FA02(x)
    print("x with FA02:", x_fa02)

    x_fa03 = FA03(x)
    print("x with FA03:", x_fa03)

    x_fa03fa03 = FA03(x_fa03)
    print("x with FA03:", x_fa03fa03)

    x_fa04 = FA04(x, FAR=2)
    print("x with FA04:", x_fa04)

