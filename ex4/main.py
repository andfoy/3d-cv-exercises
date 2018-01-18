# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import os.path as osp
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.image as mpimg


def direct_linear_transform(x1, P1, x2, P2):
    A = np.array([x1[0] * P1[2, :] - P1[0, :],
                  x1[1] * P1[2, :] - P1[1, :],
                  x2[0] * P2[2, :] - P2[0, :],
                  x2[1] * P2[2, :] - P2[1, :]])
    # print(A)
    U, S, V = np.linalg.svd(A, full_matrices=True)
    # print(Ss)
    X = V[-1, :]
    return X


def nullspace(A, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """
    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


def dist_point_line(x, l):
    """Compute distance between a line and a point."""
    a, b, c = l
    x0, y0 = x
    return np.abs(a * x0 + b * y0 + c) / np.sqrt(a**2 + b**2)


if __name__ == '__main__':
    output_path = 'output'
    if not osp.exists(output_path):
        os.makedirs(output_path)

    print('Loading images...')
    img1 = cv2.imread(osp.join('data', 'Camera00.jpg'))
    img2 = cv2.imread(osp.join('data', 'Camera01.jpg'))
    H, W, _ = img1.shape
    dims = np.array([[W, H, 1]])

    print("Loading camera parameters...")
    mat = sio.loadmat(osp.join('data', 'data.mat'))
    keys = 'K_0,K_1,R_1,t_1'
    K_0, K_1, R_1, t_1 = [mat[x] for x in keys.split(',')]
    t_1_x = np.array([[0, -t_1[:, -1], t_1[:, 1]],
                      [t_1[:, -1], 0, -t_1[:, 0]],
                      [-t_1[:, 1], t_1[:, 0], 0]])
    F = np.dot(np.dot(
        np.dot(np.linalg.inv(K_1.T), t_1_x), R_1), np.linalg.inv(K_0))

    print("Computed Fundamental Matrix")
    print(F)

    corners_c0, corners_c1 = mat['cornersCam0'], mat['cornersCam1']
    corners_c0 = corners_c0[np.argsort(corners_c0[:, 1])]
    corners_c1 = corners_c1[np.argsort(corners_c1[:, 1])]
    step = 10
    for i in range(0, corners_c0.shape[0], step):
        c0_slice = corners_c0[i: i + step, :]
        c1_slice = corners_c1[i: i + step, :]
        corners_c0[i: i + step, :] = c0_slice[np.argsort(c0_slice[:, 0])]
        corners_c1[i: i + step, :] = c1_slice[np.argsort(c1_slice[:, 0])]
    corners_c0_h = np.hstack((corners_c0, np.ones((corners_c0.shape[0], 1)))).T
    lines = np.dot(F, corners_c0_h)
    corners_c1 = corners_c1.T
    corners_c0 = corners_c0.T

    color = np.uint8([[[0, 255, 255]]])
    step = int(np.floor(179.0 / lines.shape[-1]))
    out = img2.copy()
    count = 0
    for i in range(0, lines.shape[-1]):
        line = lines[:, i]
        a, b, c = line.flatten().tolist()
        y = lambda x: int((-a / b) * (x + c / a))
        x_int = int(-c / a)
        y_int = int(-c / b)
        # x_int = int(-line[-1] / line[0])
        # y_int = (int-line[-1] / line[1])
        start = (x_int, 0) if x_int > 0 else (W, y(W))
        end = (0, y_int)
        # print([start, end])
        bgr = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)
        cv2.line(out, start, end, tuple(bgr.flatten().tolist()), 2)
        color[..., 0] += step

    # print(count)
    for i in range(0, corners_c1.shape[-1]):
        point = tuple(corners_c1[:, i].astype(np.int64).tolist())
        x1 = np.expand_dims(corners_c0[:, i], axis=1)
        x1 = np.vstack((x1, np.ones((1, 1))))
        x2 = np.expand_dims(corners_c1[:, i], axis=1)
        x2 = np.vstack((x2, np.ones((1, 1))))
        # print(np.dot(np.dot(x2.T, F), x1))
        cv2.circle(out, point, 6, (0, 255, 0), -1)

    cv2.imwrite(osp.join(output_path, 'epilines.jpg'), out)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    point_cloud = []
    color = np.uint8([[[0, 255, 255]]])
    step = int(np.floor(179.0 / lines.shape[-1]))
    composed_img = np.concatenate((img1, img2), axis=0)
    for i in range(0, lines.shape[-1]):
        line = lines[:, i].flatten().tolist()
        c0_p = tuple(corners_c0[:, i].astype(np.int64).tolist())
        min_dist = np.inf
        feature_match = None
        for j in range(1, lines.shape[-1]):
            c1_p = tuple(corners_c1[:, i].astype(np.int64).tolist())
            dist = dist_point_line(c1_p, line)
            feature_match, min_dist = (
                (c1_p, dist) if dist < min_dist else (feature_match, min_dist))
        assert feature_match is not None
        c1_d = (c1_p[0], c1_p[1] + H)
        cv2.circle(composed_img, c0_p, 6, (0, 255, 0), -1)
        cv2.circle(composed_img, c1_d, 6, (0, 255, 0), -1)
        bgr = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)
        cv2.line(composed_img, c0_p, c1_d, tuple(bgr.flatten().tolist()), 2)
        color[..., 0] += step

        X = direct_linear_transform(c0_p, K_0, c1_p, K_1)
        # print(X)
        # point_cloud.append(X)
        ax.scatter(*X.flatten().tolist())

    # print(point_cloud)
    cv2.imwrite(osp.join(output_path, 'matches.jpg'), composed_img)
    plt.show()



    # E = np.dot(np.dot(K_1.T, F), K_0)
