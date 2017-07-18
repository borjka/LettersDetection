import numpy as np
from PIL import Image
from skimage import transform, data
import math
import time


AMOUNT_OF_HOMOGRAPHIES = 10000


def random_points_for_estimation(h=32, w=32, k=0.3):
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

    index = np.random.choice(AMOUNT_OF_HOMOGRAPHIES)
    proj_trans = transform.ProjectiveTransform(matrix=all_homographies[index])
    new_pix = transform.warp(img, proj_trans)
    return new_pix


def generate_pack_of_homographies():
    homographies = np.zeros((AMOUNT_OF_HOMOGRAPHIES, 3, 3))
    for i in range(AMOUNT_OF_HOMOGRAPHIES):
        src, dst = random_points_for_estimation()
        proj_trans = transform.ProjectiveTransform()
        proj_trans.estimate(src, dst)
        homographies[i] = proj_trans.params
    np.save('homographies.npy', homographies)


def load_pack_of_homographies():
    return np.load('homographies.npy')


def add_random_blur(img, p=0.04):
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


def check_time():
    matrices = np.load('homographies.npy')
    start_time = time.time()
    for i in range(5):
        index = np.random.choice(AMOUNT_OF_HOMOGRAPHIES)
        matrix = matrices[index]
        proj_trans = transform.ProjectiveTransform(matrix=matrix)
        new_pix = transform.warp(img, proj_trans)
        Image.fromarray((new_pix * 255).astype('uint8'), 'L').show()
    print(time.time() - start_time)


def process_image(img):
    pxl = np.array(img)
    pxl = homography(pxl)
    add_random_blur(pxl)
    return pxl

all_homographies = load_pack_of_homographies()

if __name__ == '__main__':
    check_time()
