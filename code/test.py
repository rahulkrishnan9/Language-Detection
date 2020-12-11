from PIL import Image, ImageDraw, ImageFont
import numpy as np

import random
import glob
import cv2
import tensorflow as tf


def main():
    i=0
    samples = glob.glob('data/GT4HistOCR'+'/**/*.gt.txt', recursive=True)
    with open('data/sentences.txt') as f:
        my_list = f.readlines()
        for line in my_list:
            line = line.split(" ", 9)[-1].replace("|", " ")
            line = line.replace(" .", ".")
            line = line.replace(" ,", ",")
            line = line.replace(" !", "!")
            line = line.replace(" ?", "?")
            img = Image.new('RGB', (len(line)*19, 32), color = (255, 255, 255))

            fnt = ImageFont.truetype('code/courier.ttf', size=30, index=0)
            d = ImageDraw.Draw(img)
            d.text((3,5), line, font=fnt, fill=(0, 0, 0))
            img.save('data/english_generated/'+str(i) +  '.png')
            i += 1




if __name__ == '__main__':
	main()
