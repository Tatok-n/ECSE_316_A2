import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import time
import math

from transforms import fft_2d, ifft_2d, fft_1d, naive_dft_1d


def load_image(filename):

    # load image in grayscale
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: Could not load image {filename}")
        sys.exit(1)

    # resize/Pad logic
    h, w = img.shape

    # find next power of 2
    next_h = 1 << (h - 1).bit_length()
    next_w = 1 << (w - 1).bit_length()

    # pad with zeros if necessary
    if h != next_h or w != next_w:
        print(f"Resizing image from {h}x{w} to {next_h}x{next_w} (Power of 2)")
        padded_img = np.zeros((next_h, next_w))
        padded_img[:h, :w] = img
        return padded_img

    return img.astype(float)


def handle_mode_1(image):
    pass


def handle_mode_2(image):
    pass


def handle_mode_3(image):
    pass


def handle_mode_4():
    pass


def main():
    parser = argparse.ArgumentParser()

    # define arguments
    parser.add_argument(
        "-m",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Mode: [1] Fast (Default), [2] Denoise, [3] Compress, [4] Runtime",
    )
    parser.add_argument(
        "-i",
        type=str,
        default="moonlanding.png",
        help="Filename of the image to process",
    )

    args = parser.parse_args()

    # handle mode 4
    if args.mode == 4:
        handle_mode_4()
        return

    # load image for modes 1, 2, 3
    img = load_image(args.image)

    if args.mode == 1:
        handle_mode_1(img)

    elif args.mode == 2:
        handle_mode_2(img)

    elif args.mode == 3:
        handle_mode_3(img)


if __name__ == "__main__":
    main()
