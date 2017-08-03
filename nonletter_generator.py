#!/usr/bin/env python3
import time
import numpy as np
import skimage
from math import *
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from operator import itemgetter
from sys import argv


canvpath = 'canv.jpg'
canv = cv2.imread(canvpath, 1)

def rotationMatrix(angle):
    return np.array(((cos(angle), -sin(angle)),
                     (sin(angle), cos(angle))))


def drawCircle(arr, pos, color):
    x = int(round(pos[0]))
    y = int(round(pos[1]))
    if x == 32: x = 31
    if y == 32: y = 31
    arr[x, y] = color


def cvp(arr):
    return (arr[1], arr[0])


def writeSignature():
    r = (np.random.ranf((2,)) - 0.5) * np.array((0.5,0.2)) + np.array((0.5,0.15))
    v = (np.random.ranf((2,))-0.2)*0.02
    c = 0.1
    R = [r.copy(),]
    kv1 = 0.005

    for t in range(np.random.randint(500, 1500)):
        c += (np.random.ranf() - 0.5) * 0.01
        v += v * (np.random.ranf() - 0.5) * 0.02
        v = np.dot(rotationMatrix(c*0.1), v)
        if np.random.ranf() < 0.003:
            v += (np.random.ranf((2,))-0.5)*0.05
        v /= 1 + 2.*np.linalg.norm(v)
        v *= 1 + 1/(10000*np.linalg.norm(v)+1)

        v[1] = (1-kv1) * v[1] + kv1 * 0.01

        strain = copysign(max(0.3-r[0], r[0]-0.7, 0), r[0] - 0.5)
        if abs(strain) > np.random.ranf()*2. and strain * v[0] > 0:
            v[0] = -v[0]
            if np.random.ranf() > 0.5:
                c = -c
            if np.random.ranf() > 0.5:
                v[1] = -v[1]

        r += v*0.3

        for i in range(2):
            if r[i] < 0.:
                r[i] = -r[i]
                v[i] = -v[i]

            if r[i] > 1.:
                r[i] = 2. - r[i]
                v[i] = -v[i]

        R.append(r.copy())
    return R


def generate_canvas():
    Image.fromarray(np.zeros((32, 32)), 'L').save('canvas.jpg')


def generate_scribble():
    canvshape = np.array(canv.shape[:2])
    R = writeSignature()
    canvcrop = canv.copy()
    cropshape = np.array(canvcrop.shape[:2])

    for r0, r1 in zip(R[:-1], R[1:]):
        rc = (r0+r1) / 2. * cropshape
        drawCircle(canvcrop, rc, np.array((255,255,255)))

    img = cv2.cvtColor(canvcrop, cv2.COLOR_RGB2GRAY)
    # cv2.imshow('scribble', img)
    # cv2.waitKey(0)
    return img


def generate_shapes(h=32,
                    w=32,
                    letter_h=20,
                    letter_w=20,
                    N_CIRCLES=3,
                    N_LINES=6,
                    N_POLIES_MIN=3,
                    N_POLIES_MAX=15):
    img = np.zeros((h, w), np.uint8)
    N_circles = np.random.randint(N_CIRCLES)
    N_lines = np.random.randint(N_LINES)
    N_poly = np.random.randint(N_POLIES_MIN, N_POLIES_MAX)
    poly_points = []
    x0 = np.random.randint(w - letter_w)
    y0 = np.random.randint(h - letter_h)
    for _ in range(N_poly):
        x = np.random.randint(letter_w)
        y = np.random.randint(letter_h)
        poly_points.append([x0 + x, y0 + y])

    for _ in range(N_circles):
        r = np.random.randint(letter_h // 2)
        xc = np.random.randint((letter_w - 2 * r)) + r + x0
        yc = np.random.randint((letter_h - 2 * r)) + r + y0
        cv2.circle(img, (xc, yc), r, 255, 1)

    poly_points = (np.array(poly_points, np.int32)).reshape((-1, 1, 2))
    cv2.polylines(img, [poly_points], False, 255)
    # cv2.imshow('poly', img)
    # cv2.waitKey(0)

    return img


if __name__ == '__main__':
    np.random.seed()
    for _ in range(10):
        generate_shapes()
