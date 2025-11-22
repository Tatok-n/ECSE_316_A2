import numpy as np
import cmath


def dft_1d(x):
    N = len(x)
    # init output array with complex data type
    X = np.zeros(N, dtype=np.complex128)

    # iterate through each frequency component k
    for k in range(N):
        current_sum = 0
        # iterate through each time sample n
        for n in range(N):
            # discrete fourier formula
            angle = (-2j * np.pi * k * n) / N
            current_sum += x[n] * np.exp(angle)
        X[k] = current_sum

    return X


def fft_1d(x):
    N = len(x)
    # base case array of size <= 32, use regular DFT to avoid recursion overhead
    if N <= 32:
        return dft_1d(x)

    # recursion on the odd and even indexed elements
    even = fft_1d(x[0::2])
    odd = fft_1d(x[1::2])

    # compute the coefficient (twiddle factor) for odd part
    # T = exp(-2j * pi * k / N) * odd[k]
    # k = 0 to N/2 - 1
    factor = np.exp((-2j * np.pi * np.arange(N // 2)) / N)
    odd_part = factor * odd

    # combine into final fourier component
    return np.concatenate([even + odd_part, even - odd_part])


def fft_2d(img):
    pass


def ifft_2d(img_fft):
    pass


def dft_2d(img):
    pass
