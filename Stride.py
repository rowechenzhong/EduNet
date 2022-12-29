import numpy as np

from numpy.lib.stride_tricks import as_strided


# def explode2d(A, k_size, stepsize=1):
#     # Parameters
#     m, n = A.shape
#     nrows = m - k_size[0] + 1
#     ncols = n - k_size[1] + 1
#     shape = nrows, ncols, k_size[0], k_size[1]
#
#     s0, s1 = A.strides
#     strides = s0, s1, s0, s1
#
#     out_view = as_strided(A, shape=shape, strides=strides)[::stepsize, ::stepsize]
#     return out_view
#
#
# def explode3d(A, k_size, stepsize=1):
#     m, n, d = A.shape
#     nrows = m - k_size[0] + 1
#     ncols = n - k_size[1] + 1
#     shape = nrows, ncols, d, k_size[0], k_size[1]
#
#     s0, s1, s2 = A.strides
#     strides = s0, s1, s2, s0, s1
#
#     return as_strided(A, shape=shape, strides=strides)[::stepsize, ::stepsize]


def expand3d(A, k_size, stepsize=1):
    m, n, d = A.shape
    nrows = m - k_size[0] + 1
    ncols = n - k_size[1] + 1
    shape = nrows, ncols, d, k_size[0], k_size[1]

    o_rows = (m - k_size[0]) // stepsize + 1
    o_cols = (n - k_size[1]) // stepsize + 1

    s0, s1, s2 = A.strides
    strides = s0, s1, s2, s0, s1

    return as_strided(A, shape=shape, strides=strides)[::stepsize, ::stepsize].reshape((o_rows, o_cols, d, -1))


def argmax2d(A, k_size, stepsize=1):
    m, n, d = A.shape
    o_rows = (m - k_size[0]) // stepsize + 1
    o_cols = (n - k_size[1]) // stepsize + 1
    #
    # print(o_rows)
    # print(o_cols)
    # print(d)

    rows, cols = np.unravel_index(expand3d(A, k_size, stepsize).argmax(axis=3), k_size)
    row_offset = np.tile(stepsize * np.arange(o_rows).reshape((-1, 1, 1)), (1, o_cols, d))
    # print(row_offset.shape)
    col_offset = np.tile((stepsize * np.arange(o_cols)).reshape((1, -1, 1)), (o_rows, 1, d))
    rows += row_offset
    cols += col_offset
    depths = np.tile(np.arange(d), (o_rows, o_cols, 1))
    return rows, cols, depths


# if __name__ == "__main__":
# A = np.arange(70).reshape((5, 7, 2))
A = np.random.randint(0, 10, (5, 7, 2))
print(A)
print(expand3d(A, (3, 3), 2))
B = argmax2d(A, (3, 3), 2)
print(B[0].shape, B[1].shape, B[2].shape)
