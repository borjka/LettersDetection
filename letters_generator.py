import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont, ImageOps
import os
import time
import skimage
import cv2
from nonletter_generator import generate_scribble, generate_shapes


basic_path = "fonts/"
all_symbols = "123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-."
# all_symbols = "@#^.*()-'/\\|<>~"
N_symbols = len(all_symbols) + 1 # +1 if nonletters are counted


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


def add_blur(img):
    pxls = np.array(img) / 255
    pxls = skimage.util.random_noise(pxls, mode='gaussian', clip=True,
                                     mean=0, var=0.02)
    pxls = skimage.util.random_noise(pxls, mode='salt', amount=0.03)
    return pxls


def generate_nonrandom_letter(font_size=20,
                              w=32,
                              h=32,
                              bg_color=0,
                              font_color=255,
                              isToShow=False,
                              letter='A'):
    """Generate random letter, math symbol or non-letter with relatively
    random position and random font.

    Args:
        font_size: letter size
        w: width of image used to drawing letter
        h: height of image used to drawing letter
        bg_color: color of background on the image
        font_color: color of text on the image

    Returns:
        1) index of symbol in general array of symbols and 2) values of pixels in range [0, 1]
    """

    path_to_font = np.random.choice(paths_to_fonts)
    img = Image.new('L', (h, w), bg_color)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(path_to_font, size=font_size)
    letter_w, letter_h = draw.textsize(letter, font=font)
    max_x, max_y = w - letter_w - 2, h - letter_h - 2
    # pos = (x, y)
    pos = (np.random.randint(low=2, high=max_x), np.random.randint(low=2, high=max_y))
    draw.text(pos, letter, font=font, fill=font_color)
    pxls = np.array(img)
    return pxls


def add_line_to_img(img, line_type, pos, letter_w, letter_h):
    if line_type == 'S':
        delta_min = 2
        delta_max = 4
        x0 = np.random.randint(pos[0], pos[0] + letter_w + 1)
        y0 = np.random.randint(pos[1], pos[1] + letter_h + 1)
        x1 = x0 + np.random.randint(delta_min, delta_max+1)
        y1 = y0 + np.random.randint(delta_min, delta_max+1)

    if line_type == 'M':
        delta_min = 5
        delta_max = 8
        x0 = np.random.randint(pos[0], pos[0] + letter_w + 1)
        y0 = np.random.randint(pos[1], pos[1] + letter_h + 1)
        delta_x =  np.random.randint(delta_min, delta_max+1)
        delta_y =  np.random.randint(delta_min, delta_max+1)

        if x0 - pos[0] > pos[0] + letter_w - x0:
            delta_x = -delta_x

        if y0 - pos[1] > pos[1] + letter_h - y0:
            delta_y = -delta_y

        x1 = x0 + delta_x
        y1 = y0 + delta_y

    if line_type == 'L':
        delta_min = 11
        delta_max = 15
        x0 = np.random.randint(pos[0], pos[0] + letter_w + 1)
        y0 = np.random.randint(pos[1], pos[1] + round(letter_h * 0.3) + 1)
        delta_x = np.random.randint(delta_min, delta_max+1)
        delta_y = np.random.randint(delta_min, delta_max+1)

        if x0 - pos[0] > pos[0] + letter_w - x0:
            delta_x = -delta_x

        if y0 - pos[1] > pos[1] + letter_h - y0:
            delta_y = -delta_y

        x1 = x0 + delta_x
        y1 = y0 + delta_y

    cv2.line(img, (x0, y0), (x1, y1), 255, 1)


def generate_random_letter(font_size=20,
                           w=32,
                           h=32,
                           bg_color=0,
                           font_color=255,
                           isToShow=False):
    """Generate random letter, math symbol or non-letter with relatively
    random position and random font.

    Args:
        font_size: letter size
        w: width of image used to drawing letter
        h: height of image used to drawing letter
        bg_color: color of background on the image
        font_color: color of text on the image

    Returns:
        1) index of symbol in general array of symbols and 2) values of pixels in range [0, 1]
    """

    letter_index = np.random.randint(len(all_symbols))
    letter = all_symbols[letter_index]
    path_to_font = np.random.choice(paths_to_fonts)
    img = Image.new('L', (h, w), bg_color)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(path_to_font, size=font_size)
    letter_w, letter_h = draw.textsize(letter, font=font)
    max_x, max_y = w - letter_w - 2, h - letter_h - 2
    # pos = (x, y)
    pos = (np.random.randint(low=2, high=max_x), np.random.randint(low=2, high=max_y))
    draw.text(pos, letter, font=font, fill=font_color)
    pxls = np.array(img)
    for _ in range(np.random.randint(6)):
        add_line_to_img(pxls, 'S', pos, letter_w, letter_h)
    for _ in range(np.random.randint(4)):
        add_line_to_img(pxls, 'M', pos, letter_w, letter_h)
    for _ in range(np.random.randint(3)):
        add_line_to_img(pxls, 'L', pos, letter_w, letter_h)

    assert(letter_index < len(all_symbols))
    return letter_index, pxls


def generate_batch(batch_size=64, andSave=False):
    X = []
    Y = np.zeros((batch_size, N_symbols))
    for i in range(batch_size):
        isLetter = np.random.choice([True, False], p=[0.93, 0.07])
        if isLetter:
            y, x = generate_random_letter()
            x = add_blur(x)
        else:
            mode = np.random.choice([0, 1], p=[0.5, 0.5])
            if mode == 0:
                x = add_blur(generate_scribble())
            if mode == 1:
                x = add_blur(generate_shapes())
            y = N_symbols - 1

        Y[i, y] = 1
        X.append(x)
    X = np.dstack(X)
    X = np.swapaxes(X, 0, 2)
    X = np.swapaxes(X, 1, 2)
    X = X.reshape(batch_size, X.shape[1], X.shape[2], 1)
    if andSave:
        for i in range(batch_size):
            img = Image.fromarray((X[i, :, :, 0] * 255).astype('uint8'), 'L')
            index = np.argmax(Y[i])
            print(i)
            if index == N_symbols - 1:
                img.save('batch/non_letter{0}.png'.format(i))
            elif index >= 45:
                img.save('batch/math_symb{0}.png'.format(i))
            else:
                img.save('batch/'+all_symbols[index]+'{0}.png'.format(i))
    return X, Y


def main():
    for _ in range(40):
        _, pxls = generate_random_letter()
        pxls = (add_blur(pxls) * 255).astype('uint8')
        cv2.imshow('poly', pxls)
        cv2.waitKey(0)
    # Image.fromarray(pxls.astype('uint8'), 'L').show()
    # for _ in range(10):
        # s, pxls = generate_random_letter()
        # Image.fromarray(pxls.astype('uint8'), 'L').show()
        # input(all_symbols[s])

    # for _ in range(10):
        # pxls = generate_shapes()
        # Image.fromarray(pxls.astype('uint8'), 'L').show()
        # input()

    # for _ in range(10):
        # pxls = generate_scribble()
        # Image.fromarray(pxls.astype('uint8'), 'L').show()
        # input()



if __name__ == '__main__':
    main()

