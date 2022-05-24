import os
from pathlib import Path
from typing import List, Union

import numpy as np
from struct import unpack


def read_hsi(
        path: str,
) -> np.ndarray:
    """ Read HSI images (.dat files) as a numpy array (H, W, WL).
    Args:
        path: a path to a .dat file being read
    """

    if not isinstance(path, str):
        raise ValueError(f'Invalid data type: {type(path)}')
    elif not os.path.isfile(path):
        raise ValueError(f"The file doesn't exist: {path}")
    elif not path.lower().endswith('.dat'):
        raise ValueError(f"The file with extension {Path(path).suffix} is not supported at the moment")
    else:
        pass

    with open(path, 'rb') as fp:
        header = fp.read(3 * 4)
        size = list(unpack('>iii', header))
        data = np.fromfile(fp, dtype='>f')
        data = data.reshape(size)
        data = np.transpose(data, (1, 0, 2))
        data = data[::-1, ::1, :]

        return data


def get_file_list(
    src_dirs: Union[List[str], str],
    ext_list: Union[List[str], str],
    include_template: str = '',
) -> List[str]:
    """
    Get list of files with the specified extensions
    Args:
        src_dirs: directory(s) with files inside
        ext_list: extension(s) used for a search
        include_template: include directories with this template
    Returns:
        all_files: a list of file paths
    """

    all_files = []
    src_dirs = [src_dirs, ] if isinstance(src_dirs, str) else src_dirs
    ext_list = [ext_list, ] if isinstance(ext_list, str) else ext_list
    for src_dir in src_dirs:
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                file_ext = Path(file).suffix
                file_ext = file_ext.lower()
                dir_name = os.path.basename(root)
                if (
                    file_ext in ext_list
                    and include_template in dir_name
                ):
                    file_path = os.path.join(root, file)
                    all_files.append(file_path)
    all_files.sort()
    return all_files
