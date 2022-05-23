import unittest

import numpy as np
from tools.utils import read_dat


class TestReadData(unittest.TestCase):
    def test_reading(self):
        path = 'dataset/Test 1 - Liver LA contactless switching off laser/HSI/2019_08_30_10_00_39/2019_08_30_10_00_39_SpecCube.dat'
        data = read_dat(path)
        self.assertTrue(np.mean(data), float)
        self.assertEqual(round(float(np.mean(data)), 6), 0.501764)

    def test_values(self):
        self.assertRaises(ValueError, read_dat, 'unknown_dir/unknown_file.dat')

    def test_types(self):
        self.assertRaises(ValueError, read_dat, -1)
        self.assertRaises(ValueError, read_dat, 5+11j)
        self.assertRaises(ValueError, read_dat, [11, 14, 19])
        self.assertRaises(ValueError, read_dat, True)


if __name__ == '__main__':

    unittest.main()
