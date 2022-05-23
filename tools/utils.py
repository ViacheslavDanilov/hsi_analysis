import os

import numpy as np
from struct import unpack


def read_dat(
        path: str,
) -> np.ndarray:
    """ Read .dat files as a numpy array (H, W, WL).

    Args:
        path: a path to a .dat file being read
    """

    if (
            isinstance(path, str)
            and os.path.isfile(path)
    ):
        pass
    else:
        raise ValueError(f"The file doesn't exist or is not readable: {path}")

    with open(path, 'rb') as fp:
        header = fp.read(3 * 4)
        size = list(unpack('>iii', header))
        data = np.fromfile(fp, dtype='>f')
        data = data.reshape(size)
        data = np.transpose(data, (1, 0, 2))
        data = data[::-1, ::1, :]

        return data


if __name__ == '__main__':

    data_path = 'dataset/Test 1 - Liver LA contactless switching off laser/HSI/2019_08_30_10_00_39/2019_08_30_10_00_39_SpecCube.dat'
    data = read_dat(data_path)
    print(np.mean(data), np.std(data), np.min(data), np.max(data))
