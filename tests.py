import unittest
import numpy as np
from transform import fft_2d, ifft_2d, dft_2d


class TestFourierTransforms(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.size = 32
        self.image = np.random.random((self.size, self.size)) + 1j * np.random.random(
            (self.size, self.size)
        )

    def test_dft_2d(self):
        print("Testing Naive DFT 2D...", end=" ", flush=True)
        result = dft_2d(self.image)
        numpy_result = np.fft.fft2(self.image)
        self.assertTrue(np.allclose(result, numpy_result))
        print("OK")

    def test_fft_2d(self):
        print("Testing FFT 2D...", end=" ", flush=True)
        result = fft_2d(self.image)
        numpy_result = np.fft.fft2(self.image)
        self.assertTrue(np.allclose(result, numpy_result))
        print("OK")

    def test_ifft_2d(self):
        print("Testing IFFT 2D...", end=" ", flush=True)
        result = ifft_2d(self.image)
        numpy_result = np.fft.ifft2(self.image)
        self.assertTrue(np.allclose(result, numpy_result))
        print("OK")

    def test_reversibility(self):
        print("Testing Reversibility...", end=" ", flush=True)
        transformed = fft_2d(self.image)
        reconstructed = ifft_2d(transformed)
        self.assertTrue(np.allclose(self.image, reconstructed))
        print("OK")


if __name__ == "__main__":
    unittest.main()
