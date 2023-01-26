import unittest

import numpy as np

from src.data.utils import read_hsi


class TestReadData(unittest.TestCase):
    def test_reading(self):
        path = 'data/raw/30_08_2019_test_01_liver/10_00_39_T6=110/SpecCube.dat'
        data = read_hsi(path)
        self.assertTrue(np.mean(data), float)
        self.assertEqual(round(float(np.mean(data)), 6), 0.431974)

    def test_values(self):
        self.assertRaises(ValueError, read_hsi, 'unknown_dir/unknown_file.dat')
        self.assertRaises(ValueError, read_hsi, 'temp_dir/temp_file.jpg')

    def test_types(self):
        self.assertRaises(ValueError, read_hsi, -1)
        self.assertRaises(ValueError, read_hsi, 5 + 11j)
        self.assertRaises(ValueError, read_hsi, [11, 14, 19])
        self.assertRaises(ValueError, read_hsi, True)


if __name__ == '__main__':
    unittest.main()
