import unittest
import numpy as np
from tools.utils import read_hsi


class TestReadData(unittest.TestCase):
    def test_reading(self):
        path = 'dataset/Test 1 - Liver LA contactless switching off laser/HSI/2019_08_30_10_00_39/2019_08_30_10_00_39_SpecCube.dat'
        data = read_hsi(path)
        self.assertTrue(np.mean(data), float)
        self.assertEqual(round(float(np.mean(data)), 6), 0.501764)

    def test_values(self):
        self.assertRaises(ValueError, read_hsi, 'unknown_dir/unknown_file.dat')
        self.assertRaises(ValueError, read_hsi, 'temp_dir/temp_file.jpg')

    def test_types(self):
        self.assertRaises(ValueError, read_hsi, -1)
        self.assertRaises(ValueError, read_hsi, 5+11j)
        self.assertRaises(ValueError, read_hsi, [11, 14, 19])
        self.assertRaises(ValueError, read_hsi, True)


if __name__ == '__main__':

    unittest.main()
