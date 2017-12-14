# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sio


def compute_pose(K, H):
    Rt = np.dot(np.linalg.pinv(K), H)
    r1, r2 = Rt[:, 0], Rt[:, 1]
    t = Rt[:, 2]
    r1 = r1 / np.linalg.norm(r1)
    r2 = r2 / np.linalg.norm(r2)
    r3 = np.cross(r1, r2)
    R = np.vstack([r1, r2, r3]).T
    return R, t


def correct_rotation_matrix(R):
    U, _, V = np.linalg.svd(R)
    Rc = np.dot(U, V.T)
    return Rc


def compute_relative_rotation(K, H):
    R, _ = compute_pose(K, H)

    print('-' * 20)
    print('Rotation matrix estimation from homography')
    print(R)

    print('Rotation matrix properties verification')

    det = np.linalg.det(R)
    print('Determinant (Should be 1): {0}'.format(det))

    inv = np.linalg.pinv(R)
    print('Matrix inverse: ')
    print(inv)

    diff = np.linalg.norm(R.T - inv) / np.linalg.norm(R.T + inv)
    print('Absolute difference between inverse matrix '
          'and its transpose: {:.9f}'.format(diff))
    print('*' * 20)

    print(diff)
    if diff > 1e-6:
        print('Matrix is not orthogonal, correcting...')
        Rc = correct_rotation_matrix(R)

        print('Corrected matrix')
        print(Rc)

        print('Determinant: {0}'.format(np.linalg.det(Rc)))
        inv = np.linalg.pinv(Rc)
        print('Matrix inverse: ')
        print(inv)

        diff = np.linalg.norm(Rc.T - inv) / np.linalg.norm(Rc.T + inv)
        print('Absolute difference between inverse matrix '
              'and its transpose: {:.9f}'.format(diff))
    print('')


if __name__ == '__main__':
    file = sio.loadmat('data/ex2.mat')
    s, ax, ay, x0, y0 = [file[x][0][0]
                         for x in ['s', 'alpha_x', 'alpha_y', 'x_0', 'y_0']]
    K = np.array([[ax, s, x0], [0, ay, y0], [0, 0, 1]])
    H1 = file['H1']
    H2 = file['H2']
    H3 = file['H3']

    compute_relative_rotation(K, H1)
    compute_relative_rotation(K, H2)
    compute_relative_rotation(K, H3)

    R, t = compute_pose(K, H3)
    print('Rotation matrix from homography H3')
    print(R)
    print('Translation vector from homography H3')
    print(t)
