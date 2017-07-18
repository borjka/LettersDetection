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


paths_to_fonts = find_all_paths()


def generate_alphabet():
    paths_to_fonts = find_all_paths()
    for letter in all_letters:
        font = np.random.choice(paths_to_fonts)
        angle = 0
        img = Image.new('L', (h, w), bg_color)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font, size=font_size)
        letter_w, letter_h = draw.textsize(letter, font=font)
        pos = (10, 10)
        draw.text(pos, letter, font=font, fill=font_color)
        img.show()
        break


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
                           font_size=20,
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

    letter_index = np.random.randint(len(all_letters))
    letter = all_letters[letter_index]
    path_to_font = np.random.choice(paths_to_fonts)
    img = Image.new('L', (h, w), bg_color)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(path_to_font, size=font_size)
    letter_w, letter_h = draw.textsize(letter, font=font)
    pos = (3, 3)
    draw.text(pos, letter, font=font, fill=font_color)
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
    X = X.reshape(batch_size, X.shape[1], X.shape[2], 1)
    idx = [30, 12, 100]
    for i in idx:
        Image.fromarray((X[i, :, :, 0] * 255).astype('uint8'), 'L').show()
        print(all_letters[np.argmax(Y[i])])
    return X, Y

def main():
    start_time = time.time()
    generate_batch()
    print(time.time() - start_time)


if __name__ == '__main__':
    main()
