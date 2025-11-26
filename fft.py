import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2
import sys
import time


from transform import fft_2d, ifft_2d, dft_2d

# ------------------- MODE 1 (DEFAULT) CONFIGS -------------------
# USE_NUMPY : boolean (use the numpy fft or not)
USE_NUMPY = False

# ------------------- MODE 2 (DENOISING) CONFIGS -------------------

# "low_pass"  : Removing high frequencies (Keep corners)
# "high_pass" : Removing low frequencies (Remove corners)
# "magnitude" : Thresholding everything (Keep strongest coefficients)
# "hybrid"    : Keep low frequencies AND threshold the rest (Cutoff + Threshold)
DENOISE_METHOD = "hybrid"

# Parameter Value depends on method:
# For 'low_pass'/'high_pass': Fraction of width/height to keep (e.g., 0.15 = 15%)
# For 'magnitude': Percentile to CUT (e.g., 95 = keep top 5%)
# For 'hybrid': Tuple (fraction, percentile) -> (0.15, 90)
DENOISE_VALUE = (0.17, 50)

# ------------------- MODE 4 (RUNTIME ANALYSIS) CONFIGS -------------------

# Range of problem sizes (2D matrix) (Powers of 2) (e.g. 5 = 2^5 x 2^5)
# WARNING: Naive DFT is slow
RUNTIME_MIN_POWER = 5
RUNTIME_MAX_POWER = 11

# Confidence Interval for error bars (e.g., 2.0 std devs approx 95%)
CONFIDENCE_FACTOR = 2.0
NUM_TRIALS = 10


# ----------------------------- MAIN CODE -----------------------------
def load_image(filename):
    # load image in grayscale
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: Could not load image {filename}")
        sys.exit(1)

    h, w = img.shape

    # find next power of 2 to fit image
    next_h = 1 << (h - 1).bit_length()
    next_w = 1 << (w - 1).bit_length()

    # pad with zeros if necessary
    if h != next_h or w != next_w:
        padded_img = np.zeros((next_h, next_w))
        padded_img[:h, :w] = img
        return padded_img, h, w

    return img.astype(float), h, w


def plot_log_magnitude(fft_data, title="FFT Log Magnitude"):
    im = plt.imshow(np.abs(fft_data), cmap="magma", norm=LogNorm())
    plt.title(title)
    plt.colorbar(im, fraction=0.03, pad=0.03)


def handle_mode_1(image, h, w, use_numpy=False):
    # run 2D fast fourier on image
    if use_numpy:
        ft = np.fft.fft2(image)
    else:
        ft = fft_2d(image)

    # plot the result
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image[:h, :w], cmap="gray")
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plot_log_magnitude(ft, "FFT (Log Scaled)")
    plt.show()


def handle_mode_2(image, h, w, method, value):
    # setup
    ft = fft_2d(image)
    rows, cols = ft.shape
    mask = np.zeros_like(ft)

    if method == "low_pass":
        fraction = value
        r_lim = int(rows * fraction)
        c_lim = int(cols * fraction)

        # create the image mask
        mask = np.zeros_like(ft)
        mask[:r_lim, :c_lim] = 1  # keep Top-Left
        mask[:r_lim, -c_lim:] = 1  # keep Top-Right
        mask[-r_lim:, :c_lim] = 1  # keep Bottom-Left
        mask[-r_lim:, -c_lim:] = 1  # keep Bottom-Right

    elif method == "high_pass":
        fraction = value
        r_lim = int(rows * fraction)
        c_lim = int(cols * fraction)

        mask[:] = 1  # start with all 1s

        # Remove corners
        mask[:r_lim, :c_lim] = 0
        mask[:r_lim, -c_lim:] = 0
        mask[-r_lim:, :c_lim] = 0
        mask[-r_lim:, -c_lim:] = 0

    elif method == "magnitude":
        percentile = value
        # treshold magnitude
        thresh = np.percentile(np.abs(ft), percentile)
        # keep coefficients stronger than threshold
        mask = np.abs(ft) >= thresh

    elif method == "hybrid":
        # value should be a tuple: (fraction, percentile)
        fraction, percentile = value

        # Low Pass Mask
        r_lim = int(rows * fraction)
        c_lim = int(cols * fraction)
        spatial_mask = np.zeros_like(ft)
        spatial_mask[:r_lim, :c_lim] = 1
        spatial_mask[:r_lim, -c_lim:] = 1
        spatial_mask[-r_lim:, :c_lim] = 1
        spatial_mask[-r_lim:, -c_lim:] = 1

        # Magnitude Mask
        thresh = np.percentile(np.abs(ft), percentile)
        mag_mask = np.abs(ft) >= thresh

        # combine the masks
        mask = np.logical_or(spatial_mask, mag_mask).astype(float)

    # apply mask
    ft_denoised = ft * mask

    # output stats
    non_zeros = np.count_nonzero(ft_denoised)
    total = rows * cols
    fraction = non_zeros / total
    print(f"Non-zeros: {non_zeros}")
    print(f"Fraction of coefficients kept: {fraction:.2%}")

    # get image back
    img_denoised = np.abs(ifft_2d(ft_denoised))

    # plot original and denoised
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image[:h, :w], cmap="gray")
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(img_denoised[:h, :w], cmap="gray")
    plt.title(f"Denoised: {method} ({value})")

    plt.show()


def handle_mode_3(image, h, w):
    # setup
    ft = fft_2d(image)
    compression_levels = [0, 50, 80, 95, 98, 99.9]
    plt.figure(figsize=(12, 8))

    # process each compression levels
    for i, p in enumerate(compression_levels):
        # find the treshold (minimum to keep a coefficient)
        threshold = np.percentile(np.abs(ft), p)

        # apply mask to keep only those coefficient
        mask = np.abs(ft) >= threshold
        ft_compressed = ft * mask

        # get back image and compute stats
        img_compressed = np.abs(ifft_2d(ft_compressed))

        nz = np.count_nonzero(mask)
        size_bytes = nz * 16  # complex128 takes 16 bytes
        print(
            f"Level {i+1}: {p}% compressed. Non-zeros: {nz}. Approx Size: {size_bytes/1024:.1f} KB"
        )

        # plot it
        plt.subplot(2, 3, i + 1)
        plt.imshow(img_compressed[:h, :w], cmap="gray")
        plt.title(f"{p}% Compression")
        plt.axis("off")

    plt.show()


def handle_mode_4(min_pow, max_pow, trials, conf_factor):
    sizes = []
    dft_means = []
    dft_stds = []
    fft_means = []
    fft_stds = []

    # Loop through powers of 2
    for p in range(min_pow, max_pow + 1):
        N = 2**p
        sizes.append(N)

        print(f"\n--- Processing Matrix Size {N}x{N} (2^{p}) ---")

        # generate random 2D complex array
        image = np.random.random((N, N)) + 1j * np.random.random((N, N))

        t_dft = []
        t_fft = []

        # run Trials
        for t in range(trials):
            print(f"    Running Trial {t + 1}/{trials}...", end="\r", flush=True)

            # measure regular dft
            start = time.time()
            dft_2d(image)
            t_dft.append(time.time() - start)

            # measure fft
            start = time.time()
            fft_2d(image)
            t_fft.append(time.time() - start)

        # calculate Stats
        nm = np.mean(t_dft)
        nv = np.var(t_dft)
        ns = np.std(t_dft)

        fm = np.mean(t_fft)
        fv = np.var(t_fft)
        fs = np.std(t_fft)

        dft_means.append(nm)
        dft_stds.append(ns)
        fft_means.append(fm)
        fft_stds.append(fs)

        print(f"    [Naive DFT]  Mean: {nm:.5f}s | Variance: {nv:.5e}")
        print(f"    [FFT] Mean: {fm:.5f}s | Variance: {fv:.5e}")

    # Plot
    plt.figure(figsize=(10, 6))

    # Plot DFT
    plt.errorbar(
        sizes,
        dft_means,
        yerr=np.array(dft_stds) * conf_factor,
        fmt="-o",
        label="Naive 2D DFT",
        capsize=5,
    )

    # Plot FFT
    plt.errorbar(
        sizes,
        fft_means,
        yerr=np.array(fft_stds) * conf_factor,
        fmt="-s",
        label="Fast Fourier Transform (2D)",
        capsize=5,
    )
    plt.xscale("log", base=2)
    plt.xlabel("Problem Size (NxN)")
    plt.yscale("log", base=10)
    plt.ylabel("Runtime (seconds)")
    plt.title(f"Runtime Comparison: Naive vs FFT (Error Bars = {conf_factor}*std)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)

    plt.show()


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
    if args.m == 4:
        handle_mode_4(
            RUNTIME_MIN_POWER, RUNTIME_MAX_POWER, NUM_TRIALS, CONFIDENCE_FACTOR
        )
        return

    # load image for modes 1, 2, 3
    img, h, w = load_image(args.i)

    if args.m == 1:
        handle_mode_1(img, h, w, USE_NUMPY)

    elif args.m == 2:
        handle_mode_2(img, h, w, DENOISE_METHOD, DENOISE_VALUE)

    elif args.m == 3:
        handle_mode_3(
            img,
            h,
            w,
        )


if __name__ == "__main__":
    main()
