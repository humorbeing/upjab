import numpy as np

def getI(str_list="00123"):
    temp = [int(s) for s in str_list]
    temp.sort()
    return temp


def getS(int_list=[0, 1, 2, 3]):
    int_list.sort()
    temp = "".join([str(i) for i in int_list])
    return temp

from collections import OrderedDict


def getOrder_norepeat(n=3, FAR=2):  # number of original features  # feature augmentation runs
    index_list = []
    for i in range(n):
        index_list.append(getS([i]))

    for _ in range(FAR):
        index_list1 = []
        for i in range(len(index_list)):
            for j in range(i+1, len(index_list)):
                temp11 = index_list[i] + index_list[j]
                index_list1.append(getS(getI(temp11)))

        index_list = index_list + index_list1
        index_list = list(OrderedDict.fromkeys(index_list))

    # index_list.sort()
    return index_list


def getOrder_full(n=3, FAR=2):  # number of original features  # feature augmentation runs
    index_list = []
    for i in range(n):
        index_list.append(getS([i]))

    for _ in range(FAR):
        index_list1 = []
        for i in range(len(index_list)):
            for j in range(len(index_list)):
                temp11 = index_list[i] + index_list[j]
                index_list1.append(getS(getI(temp11)))

        index_list = index_list + index_list1
        index_list = list(OrderedDict.fromkeys(index_list))

    # index_list.sort()
    return index_list

def full_cross_multiply_norepeat(x, FAR=2):

    n_samples, n_features = x.shape
    index_list = getOrder_norepeat(n=n_features, FAR=FAR)

    x_new = np.zeros((n_samples, len(index_list)))
    for i in range(len(index_list)):
        index_order = index_list[i]
        index_order_list = getI(index_order)
        new_feature = np.ones((n_samples, 1))
        for index in index_order_list:
            new_feature *= x[:, index].reshape(-1, 1)
        x_new[:, i] = new_feature.flatten()

    return x_new



def full_cross_multiply_full(x, FAR=2):

    n_samples, n_features = x.shape
    index_list = getOrder_full(n=n_features, FAR=FAR)

    x_new = np.zeros((n_samples, len(index_list)))
    for i in range(len(index_list)):
        index_order = index_list[i]
        index_order_list = getI(index_order)
        new_feature = np.ones((n_samples, 1))
        for index in index_order_list:
            new_feature *= x[:, index].reshape(-1, 1)
        x_new[:, i] = new_feature.flatten()

    return x_new

if __name__ == "__main__":
    feature_list = [
        [1, 2, -3],
        [4, 5, 6],
        [-7, 8, 9]
    ]
    feature_list = np.array(feature_list)
    print('====================')
    new_feature = full_cross_multiply_norepeat(feature_list, FAR=2)
    print(new_feature)
    new_feature = full_cross_multiply_full(feature_list, FAR=2)
    print(new_feature)
    print("done")
