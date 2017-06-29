import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageDraw, ImageFont, ImageOps
import os
import time
import homography


basic_path = "fonts/"
all_letters = "abcdefghijklmnopqrstuvwxyzABDEFGHIJKLMNQRTVYZ"


def find_all_paths():
    fonts_names = os.listdir(basic_path)
    if fonts_names[0][0] == '.':
        del fonts_names[0]
    paths_to_fonts = []
    for font_name in fonts_names:
        for file in os.listdir(basic_path+font_name):
            if isFontFormat(file):
                paths_to_fonts.append(basic_path+font_name+"/"+file)
    return paths_to_fonts


def isFontFormat(file_name):
    if file_name.endswith(".ttf") or file_name.endswith(".otf"):
        return True
    else:
        return False


def pos_based_on_alpha(w, h, img_w, img_h, alpha):
    """Find the best position of letter on the image
    based on angle of rotation

    Args:
        w, h: width and height of drawing letter
        img_w, img_h: width and height of image
        alpha: angle of letter's rotation

    Return:
        Best position
    """

    cos_a, sin_a = math.cos(math.radians(abs(alpha))), math.sin(math.radians(abs(alpha)))
    new_w = cos_a * w + sin_a * h
    new_h = sin_a * w + cos_a * h
    l_x_pos = abs(new_w - w)
    r_x_pos = img_w - max(new_w, w)
    delta_x = r_x_pos - l_x_pos
    t_y_pos = abs(new_h - h)
    b_y_pos = img_h - max(new_h, h)
    delta_y = b_y_pos - t_y_pos
    all_k_x = [0.2, 0.4, 0.6, 0.8]
    all_k_y = [0.2, 0.4, 0.6, 0.8]
    k_x = np.random.choice(all_k_x)
    k_y = np.random.choice(all_k_y)

    x = round(l_x_pos + k_x * delta_x)
    y = round(t_y_pos + k_y * delta_y)

    return (x, y)


def generate_random_letter(letter="w",
                           font_size=14,
                           w=32,
                           h=32,
                           bg_color=0,
                           font_color=255,
                           nonLetter=False
                           ):
    """Creates samples for one letter with all possible fonts,
    rotations and different positions on the image.

    Args:
        letter: letter to be drawn
        font_size: letter size
        w: width of image used to drawing letter
        h: height of image used to drawing letter
        bg_color: color of background on the image
        font_color: color of text on the image

    Returns:
        3-D np.array containing images with pixels normalised to be in range of [0, 1]
    """
    paths_to_fonts = find_all_paths()
    angles = [-10, -5, 0, 5, 10]
    letter_index = np.random.randint(len(all_letters))
    letter = all_letters[letter_index]
    font = np.random.choice(paths_to_fonts)
    angle = np.random.choice(angles)
    img = Image.new('L', (h, w), bg_color)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font, size=font_size)
    letter_w, letter_h = draw.textsize(letter, font=font)
    pos = pos_based_on_alpha(letter_w, letter_h, w, h, angle)
    draw.text(pos, letter, font=font, fill=font_color)
    img = img.rotate(angle)
    return letter_index, homography.process_image(img)


def generate_batch(batch_size=128):
    X = []
    Y = np.zeros((batch_size, len(all_letters)))
    for i in range(batch_size):
        y, x = generate_random_letter()
        Y[i, y] = 1
        X.append(x)
    X = np.dstack(X)
    X = np.swapaxes(X, 0, 2)
    X = np.swapaxes(X, 1, 2)
    idx = [12, 53, 100]
    for i in idx:
        Image.fromarray((X[i] * 255).astype('uint8'), 'L').show()
        print(all_letters[np.argmax(Y[i])])

def main():
    generate_batch()


if __name__ == '__main__':
    main()
