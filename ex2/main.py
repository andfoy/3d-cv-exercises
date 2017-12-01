# -*- coding: utf-8 -*-

import os
import cv2
import glob
import argparse
import numpy as np
import os.path as osp
import scipy.io as io


parser = argparse.ArgumentParser(
    description='Project a 3D world coordinate into 2D image'
                'pixel coordinates')

parser.add_argument('--correction', action='store_true',
                    help='apply radial distortion correction')

args = parser.parse_args()

color = (0, 0, 255)
if args.correction:
    color = (0, 255, 0)

def project_points(X, K, R, T, distortion_params=None):
    """
    Project points from 3d world coordinates to 2d image coordinates
    """
    x_2d = np.dot(K, (np.dot(R, X) + T))
    x_2d = x_2d[:-1, :] / x_2d[-1, :]
    if distortion_params is not None:
        x_2d_norm = np.concatenate((x_2d, np.ones((1, x_2d.shape[1]))), 0)
        x_3d_norm = np.dot(np.linalg.pinv(K), x_2d_norm)
        x_2d_post = x_3d_norm[:-1, :] / x_3d_norm[-1, :]
        r = np.sqrt(x_2d_post[0, :]**2 + x_2d_post[1, :]**2)
        correction = (1 + distortion_params[0] * r**2 +
                      distortion_params[1] * r**4 +
                      distortion_params[4] * r**6)
        x_2d_corr = x_2d_post * correction
        x_3d_corr = np.concatenate((
            x_2d_corr, np.ones((1, x_2d_corr.shape[1]))), 0)
        x_2d = np.dot(K, x_3d_corr)
        x_2d = x_2d[:-1, :] / x_2d[-1, :]
    return x_2d

def project_and_draw(img_path, X, K, R, T, distortion_parameters=None):
    """
    Draw projected points on an image.
    """
    img = cv2.imread(img_path)
    img_name, _ = osp.splitext(osp.basename(img_path))
    points = project_points(
        X, K, R, T, distortion_params=distortion_parameters)
    points = np.round(points.T).astype(np.int64).tolist()
    for point in points:
        cv2.circle(img, tuple(point), 2, color, -1)
    output_file_name = '{0}_proj'
    if args.correction:
        output_file_name += '_correction'
    output_file_name += '.jpg'
    cv2.imwrite(osp.join(
        'results', output_file_name.format(img_name)), img)


if __name__ == '__main__':
    base_folder = osp.join('.', 'data')
    if not osp.exists('results'):
        os.makedirs('results')

    image_num = 0
    data = io.loadmat(osp.join(base_folder, 'ex1.mat'))

    """
    Intrinsic parameters.
    """
    # Distortion parameters
    kc = data['dist_params'] if args.correction else None
    # K matrix of the cameras
    K = data['intinsic_matrix']

    for image in glob.glob(osp.join(base_folder, '*.jpg')):
        print('Processing image: {0}'.format(image))
        image_num = int(osp.splitext(osp.basename(image))[0])

        X = data['X_3D'][image_num]
        """
        Translation vector: as the world origin is seen
        from the camera coordinates
        """
        T = data['TVecs'][image_num]
        """
        Rotation matrices: converts coordinates from world to camera
        """
        R = data['RMats'][image_num]
        # imgs = [cv.imread(base_folder + str(i).zfill(5) + '.jpg')
        #         for i in range(TVecs.shape[0])]

        project_and_draw(image, X, K, R, T, kc)
