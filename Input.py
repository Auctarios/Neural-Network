import numpy as np
# from Layer import *
# class Input(Layer):
a = np.array([[[1,2,3],
     [3,4,5],
     [5,6,7]],
     [[1,2,3],
     [3,4,5],
     [5,6,7]]])

# print(a)

def flatten(arr = None):
    if arr is None:
        raise ValueError("Nothing to flatten.")
    # print(len(arr.shape))
    if len(arr.shape) > 2:
        shape = 1
        for i, sp in enumerate(arr.shape):
            # print(i, sp, shape)
            if i != 0:
                shape *= sp
        print(shape)
        for a in arr:
            print(a, a.shape)
            print("-----------")
            a.reshape(-1, shape)
            print(a, a.shape)
            print("-----------")
    return arr

print(flatten(a))

# b = np.array([[1,2,3],[1,2,3],[1,2,3]])
# print(b)
# print(b.reshape(-1,9))