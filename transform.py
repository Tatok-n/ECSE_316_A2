import numpy as np


def dft_1d(x):
    # X_k = sum(x_n * exp((-2j*pi*k*n)/N) )
    # Notice that Discrete fourier formula is the dot product of x_n and the exponential terms

    # make x a vector
    x = np.asarray(x, dtype=np.complex128)

    # get vector width
    N = x.shape[0]

    n = np.arange(N)  # create vector 0 to N-1
    k = n.reshape((N, 1))  # turn into column vector

    # calculate all the exponential terms as a matrix M using nxk matrix multiplication
    M = np.exp((-2j * np.pi * k * n) / N)

    # dft =  x dot M
    return np.dot(x, M)


def dft_2d(img):
    # apply dft on rows (axis 1)
    rows_result = np.apply_along_axis(dft_1d, axis=1, arr=img)
    # apply dft on columns (axis 0)
    cols_result = np.apply_along_axis(dft_1d, axis=0, arr=rows_result)
    return cols_result


def fft_1d(x):
    # make x a vector
    x = np.asarray(x, dtype=np.complex128)

    # get vector width
    N = x.shape[0]

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
    # apply fft on rows (axis 1)
    rows_result = np.apply_along_axis(fft_1d, axis=1, arr=img)
    # apply fft on columns (axis 0)
    cols_result = np.apply_along_axis(fft_1d, axis=0, arr=rows_result)
    return cols_result


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
