import numpy as np
from PIL import Image
from skimage import transform, data
import math
import time


def light_random_points_for_estimation(h=32, w=32, k=0.1):
    dx = w * k
    dy = h * k
    src = [[0, 0], [h, 0], [0, w], [h, w]]
    dst = [[], [], [], []]
    for p in range(4):
        dst[p].append(src[p][0] + np.random.randint(low=-dy, high=dy))
        dst[p].append(src[p][1] + np.random.randint(low=-dx, high=dx))
    return np.array(src), np.array(dst)


def generate_transformation_matrix(isTFVersion=True):
    src, dst = light_random_points_for_estimation()
    proj_trans = transform.ProjectiveTransform()
    proj_trans.estimate(src, dst)
    if isTFVersion:
        return np.concatenate((proj_trans.params[0], proj_trans.params[1], proj_trans.params[2][:2]))
    return proj_trans.params


def apply_homography(img):
    proj_trans = transform.ProjectiveTransform(matrix=generate_transformation_matrix())
    new_pix = transform.warp(img, proj_trans)
    return new_pix


if __name__ == '__main__':
    generate_transformation_matrix()
