import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont, ImageOps
import os
import time
import skimage
from nonletter_generator import generate_scribble, generate_shapes


basic_path = "fonts/"
all_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-.|\'"
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
    if letter_index >= N_symbols:
        img = img.rotate(180)
    pxls = np.array(img) / 255
    pxls = skimage.util.random_noise(pxls, mode='gaussian', clip=True,
                                     mean=0, var=0.02)
    pxls = skimage.util.random_noise(pxls, mode='salt', amount=0.02)
    if isToShow:
        Image.fromarray((pxls * 255).astype('uint8'), 'L').show()
    if letter_index >= N_symbols:
        return N_symbols, pxls
    return letter_index, pxls


def generate_batch(batch_size=32, andSave=False):
    X = []
    Y = np.zeros((batch_size, N_symbols))
    for i in range(batch_size):
        isLetter = np.random.choice([True, False], p=[0.95, 0.05])
        if isLetter:
            y, x = generate_random_letter()
        else:
            mode = np.random.choice([0, 1], p=[0.5, 0.5])
            if mode == 0:
                x = generate_scribble()
            if mode == 1:
                x = generate_shapes()
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
    X, Y = generate_batch(andSave=True)
    print(X.shape, Y.shape)


if __name__ == '__main__':
    main()
