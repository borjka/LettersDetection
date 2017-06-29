import numpy as np
from PIL import Image
from skimage import transform, data
import math
import time

def random_points_for_estimation(h=32, w=32, k=0.2):
    dx = h * k
    dy = w * k
    src = [[0, 0], [h, 0], [0, w], [h, w]]
    dst = [[], [], [], []]
    mode = [[], [], [], []]
    for i in range(4):
        if i % 2 == 0:
            mode[i].append(np.random.choice([-1, 1]))
        else:
            mode[i].append(-mode[i-1][0])

    for i in range(4):
        if i  == 0 or i == 1:
            mode[i].append(np.random.choice([-1, 1]))
        else:
            mode[i].append(-mode[i-2][1])

    for dim in range(2):
        for i, src_p in enumerate(src):
            dst[i].append(src_p[dim] + mode[i][dim] * np.random.randint(dx))

    return np.array(src), np.array(dst)


def homography(img):
    """Make some random projective transformation(homography)
    on the input image.

    Args:
        img: np.array containing values of pixels
    Returns:
        New np.array containing processed pixels.
    """

    src, dst = random_points_for_estimation(h=img.shape[0], w=img.shape[1])
    proj_trans = transform.ProjectiveTransform()
    proj_trans.estimate(src, dst)
    new_pix = transform.warp(img, proj_trans)
    return new_pix


def add_random_blur(img, p=0.02):
    """Add some random pixels to the image with some probability.

    Args:
        img: np.array containing values of pixels.
        p: probability of pixels to be changed.
    """

    values = [False, True]
    probabilities = [1-p, p]
    for row  in range(img.shape[0]):
        for col in range(img.shape[1]):
            isChangeColor = np.random.choice(a=values, p=probabilities)
            if isChangeColor:
                img[row, col] = np.random.rand()


def process_image(img):
    pxl = np.array(img)
    pxl = homography(pxl)
    add_random_blur(pxl)
    # Image.fromarray((pxl * 255).astype('uint8'), 'L').show()
    return pxl


if __name__ == '__main__':
   generate_img()
