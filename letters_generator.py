import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageDraw, ImageFont, ImageOps
import os
import time

def isFontFormat(file_name):
    if file_name.endswith(".ttf") or file_name.endswith(".otf"):
        return True
    else:
        return False

basic_path = "fonts/"
fonts_names = os.listdir(basic_path)
if fonts_names[0][0] == '.':
    del fonts_names[0]
paths_to_fonts = []
for font_name in fonts_names:
    for file in os.listdir(basic_path+font_name):
        if isFontFormat(file):
            paths_to_fonts.append(basic_path+font_name+"/"+file)

def check_all_fonts(part=0):
    font_size=14
    w=32
    h=32
    bg_color=0
    font_color=255
    k = 20
    letter = 'f'
    img = Image.new('L', (h, w), bg_color)
    draw = ImageDraw.Draw(img)
    for path_to_font in paths_to_fonts[part*k:(part+1)*k]:
        font = ImageFont.truetype(path_to_font, size=font_size)
        letter_w, letter_h = draw.textsize(letter, font=font)
        pos = pos_based_on_alpha(letter_w, letter_h, w, h, 0)
        for p in pos[:1]:
            img = Image.new('L', (h, w), bg_color)
            draw = ImageDraw.Draw(img)
            draw.text(p, letter, font=font, fill=font_color)
            img.show()
            print(path_to_font)
        input("Press key to continue")



def pos_based_on_alpha(w, h, img_w, img_h, alpha):
    cos_a, sin_a = math.cos(math.radians(abs(alpha))), math.sin(math.radians(abs(alpha)))
    new_w = cos_a * w + sin_a * h
    new_h = sin_a * w + cos_a * h
    l_x_pos = abs(new_w - w)
    r_x_pos = img_w - max(new_w, w)
    delta_x = r_x_pos - l_x_pos
    t_y_pos = abs(new_h - h)
    b_y_pos = img_h - max(new_h, h)
    delta_y = b_y_pos - t_y_pos
    all_k_x = [0, 0.2, 0.4, 0.6, 0.8, 1]
    all_k_y = [0, 0.2, 0.4, 0.6, 0.8, 1]
    all_pos = []

    for k_y in all_k_y:
        for k_x in all_k_x:
            x = round(l_x_pos + k_x * delta_x)
            y = round(t_y_pos + k_y * delta_y)
            pos = (x, y)
            all_pos.append(pos)

    return all_pos

def letter_with_fonts(letter="w",
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
    angles = [-15, -10, -5, 0, 5, 10, 15]
    all_fonts = []
    img = Image.new('L', (h, w), bg_color)
    draw = ImageDraw.Draw(img)
    for path_to_font in paths_to_fonts:
        font = ImageFont.truetype(path_to_font, size=font_size)
        letter_w, letter_h = draw.textsize(letter, font=font)
        for angle in angles:
            pos = pos_based_on_alpha(letter_w, letter_h, w, h, angle)
            for p in pos:
                img = Image.new('L', (h, w), bg_color)
                draw = ImageDraw.Draw(img)
                draw.text(p, letter, font=font, fill=font_color)
                img = img.rotate(angle)
                if nonLetter:
                    img = img.rotate(180)

                pixels = np.array(img)
                pixels = pixels / 255.0
                all_fonts.append(pixels)
    return np.dstack(all_fonts)


def save_letter(letter="a", path="imgs_of_letters/", nonLetter=False):
    pix = letter_with_fonts(letter=letter, nonLetter=nonLetter)
    pix = np.swapaxes(pix, 0, 2)
    pix = np.swapaxes(pix, 1, 2)
    np.random.shuffle(pix)
    n_parts = 8
    imgs_for_batch = pix.shape[0] // n_parts

    for i in range(n_parts):
        if nonLetter:
            if letter.istitle():
                np.save(path+str(i)+"/non_big_"+letter.lower(),
                    pix[i * imgs_for_batch : (i+1) * imgs_for_batch])
            else:
                np.save(path+str(i)+"/non_"+letter,
                    pix[i * imgs_for_batch : (i+1) * imgs_for_batch])

        else:
            if letter.istitle():
                np.save(path+str(i)+"/big_"+letter.lower(),
                    pix[i * imgs_for_batch : (i+1) * imgs_for_batch])
            else:
                np.save(path+str(i)+"/"+letter,
                    pix[i * imgs_for_batch : (i+1) * imgs_for_batch])

    print("Letter \'"+letter+"\' saved!")


def test():
    check_all_fonts(part=8)

def main():

    all_letters = "abcdefghijklmnopqrstuvwxyzABDEFGHIJKLMNQRTVYZ"
    non_letters = "fgijkyhrFYR"
    for letter in non_letters:
        save_letter(letter=letter, nonLetter=True)


if __name__ == '__main__':
    main()
