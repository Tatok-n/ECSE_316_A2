import numpy as np


def dft_1d(x):
    # X_k = sum(x_n * exp((-2j*pi*k*n)/N) )
    # Notice that Discrete fourier formula is the dot product of x_n and the exponential terms

    # make x a vector
    x = np.asarray(x, dtype=np.complex128)

    # get vector width
    N = x.shape[-1]

    n = np.arange(N)  # create vector 0 to N-1
    k = n.reshape((N, 1))  # turn into column vector

    # calculate all the exponential terms as a matrix M using nxk matrix multiplication
    M = np.exp((-2j * np.pi * k * n) / N)

    # dft =  x dot M
    return np.dot(x, M)


def dft_2d(img):
    # apply dft on the rows
    rows_result = dft_1d(img)

    # transpose it
    rows_result_T = rows_result.T

    # apply dft on the colums
    columns_result = dft_1d(rows_result_T)

    # retranspose to get original orientation
    return columns_result.T


def fft_1d(x):
    N = len(x)
    # base case array of size <= 4, use regular DFT to avoid recursion overhead
    if N <= 4:
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
    h, w = img.shape

    # FFT the rows
    rows_result = np.zeros((h, w), dtype=np.complex128)
    for i in range(h):
        rows_result[i] = fft_1d(img[i])

    # FFT the columns
    # transpose so columns becomes rows and we can use fft_1d
    transposed_img = rows_result.T
    cols_result = np.zeros((w, h), dtype=np.complex128)

    for i in range(w):
        cols_result[i] = fft_1d(transposed_img[i])

    # retranspose the result to get the result
    return cols_result.T


def ifft_2d(img_fft):
    # Use the property: IFFT(X) = conj(FFT(conj(X))) / (N*M)
    h, w = img_fft.shape

    # conjugate image
    img_conj = np.conjugate(img_fft)

    # apply fft
    result_conj = fft_2d(img_conj)

    # conjugate results and divide by N*M
    result = np.conjugate(result_conj) / (h * w)

    return result
