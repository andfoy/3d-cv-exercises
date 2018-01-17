# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import os.path as osp
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


if __name__ == '__main__':
    output_path = 'output'
    if not osp.exists(output_path):
        os.makedirs(output_path)

    print('Loading images...')
    img1 = mpimg.imread(osp.join('data', 'Camera00.jpg'))
    img2 = mpimg.imread(osp.join('data', 'Camera01.jpg'))
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
    l = np.dot(F, corners_c0_h)
    corners_c1 = corners_c1.T
    corners_c0 = corners_c0.T

    color = np.uint8([[[0, 255, 255]]])
    step = int(np.floor(179.0 / l.shape[-1]))
    out = img2.copy()
    count = 0
    for i in range(0, l.shape[-1]):
        line = l[:, i]
        a, b, c = line.flatten().tolist()
        y = lambda x: int((-a / b) * (x + c / a))
        x_int = int(-c / a)
        y_int = int(-c / b)
        # x_int = int(-line[-1] / line[0])
        # y_int = (int-line[-1] / line[1])
        start = (x_int, 0) if x_int > 0 else (W, y(W))
        end = (0, y_int)
        print([start, end])
        bgr = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)
        cv2.line(out, start, end, tuple(bgr.flatten().tolist()), 2)
        color[..., 0] += step

    print(count)
    for i in range(0, corners_c1.shape[-1]):
        point = tuple(corners_c1[:, i].astype(np.int64).tolist())
        x1 = np.expand_dims(corners_c0[:, i], axis=1)
        x1 = np.vstack((x1, np.ones((1, 1))))
        x2 = np.expand_dims(corners_c1[:, i], axis=1)
        x2 = np.vstack((x2, np.ones((1, 1))))
        # print(np.dot(np.dot(x2.T, F), x1))
        cv2.circle(out, point, 6, (0, 255, 0), -1)

    cv2.imwrite(osp.join(output_path, 'epipolar_lines_cam0.jpg'), out)
