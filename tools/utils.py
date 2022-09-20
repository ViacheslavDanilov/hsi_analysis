import os
import re
import logging
from pathlib import Path
from typing import List, Union, Tuple, Optional

import cv2
import numpy as np
from glob import glob
from struct import unpack


def read_hsi(
        path: str,
        data_type: str = 'absorbance',
) -> np.ndarray:
    """
    Read HSI images (.dat files) as a numpy array (H, W, WL)

    Args:
        path: a path to a .dat file being read
        data_type: a mode of reading HSI images
    Returns:
        data: HSI represented in a 3D NumPy array
    """

    if not isinstance(path, str):
        raise ValueError(f"Invalid data type: {type(path)}")
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

        if data_type == 'reflectance':
            pass
        elif data_type == 'absorbance':
            data = -np.log10(np.maximum(data, 0.01))
        else:
            raise ValueError(f'Invalid data type: {data_type}')

        data = data.astype(np.float32)

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


def get_dir_list(
    data_dir: str,
    include_dirs: Optional[Union[List[str], str]] = None,
    exclude_dirs: Optional[Union[List[str], str]] = None,
) -> List[str]:
    """
    Filter the list of studied directories

    Args:
        data_dir: source directory with subdirectories
        include_dirs: directories to be included
        exclude_dirs: directories to be excluded

    Returns:
        dir_list: filtered list of studied directories
    """

    include_dirs = [include_dirs, ] if isinstance(include_dirs, str) else include_dirs
    exclude_dirs = [exclude_dirs, ] if isinstance(exclude_dirs, str) else exclude_dirs

    dir_list = []
    _dir_list = glob(data_dir + '/*/')
    for studied_dir in _dir_list:
        if include_dirs and Path(studied_dir).name not in include_dirs:
            logging.info(
                'Skip {:s} because it is not in the include_dirs list'.format(
                    Path(studied_dir).name
                )
            )
            continue

        if exclude_dirs and Path(studied_dir).name in exclude_dirs:
            logging.info(
                'Skip {:s} because it is in the exclude_dirs list'.format(
                    Path(studied_dir).name
                )
            )
            continue

        dir_list.append(studied_dir)
    return dir_list


def extract_body_part(
        path: str,
) -> str:
    """
    Extract a body part name based on the HSI path

    Args:
        path: a path to a file
    Returns:
        body_part:
    """

    if 'liver' in path.lower():
        body_part = 'Liver'
    elif 'pancreas' in path.lower():
        body_part = 'Pancreas'
    elif 'stomach' in path.lower():
        body_part = 'Stomach'
    elif 'heart' in path.lower():
        body_part = 'Heart'
    else:
        body_part = 'Unknown'

    return body_part


def extract_time_stamp(
        filename: str,
) -> Tuple[str, str]:
    """
    Extract a time stamp from the HSI filename

    Args:
        filename: a filename with an unstructured time stamp
    Returns:
        date: a date string in format DD.MM.YYYY
        time: a time string in format HH:MM:SS
    """

    try:
        datetime_list = re.findall(r'\d+', filename)
        if len(datetime_list) == 6:
            date = f'{datetime_list[2]}.{datetime_list[1]}.{datetime_list[0]}'
            time = f'{datetime_list[3]}:{datetime_list[4]}:{datetime_list[5]}'
        else:
            date = 'Invalid date'
            time = 'Invalid time'
    except Exception as e:
        date = 'Invalid date'
        time = 'Invalid time'

    return date, time


def get_color_map(
        color_map: str,
):
    if color_map == 'jet':
        cmap = cv2.COLORMAP_JET
    elif color_map == 'bone':
        cmap = cv2.COLORMAP_BONE
    elif color_map == 'ocean':
        cmap = cv2.COLORMAP_OCEAN
    elif color_map == 'cool':
        cmap = cv2.COLORMAP_COOL
    elif color_map == 'hsv':
        cmap = cv2.COLORMAP_HSV
    else:
        raise ValueError(f'Unknown color map: {color_map}')
    return cmap


def crop_image(
        input_img: np.ndarray,
        img_type: str = 'absorbance',
) -> np.ndarray:

    assert input_img.shape[1] % 2 == 0, 'Input image width should be divisible by 2 (contain 2 sub-images)'

    img_width = int(input_img.shape[1] / 2)
    if img_type == 'absorbance':
        idx = 0
    elif img_type == 'hsv':
        idx = 1
    elif img_type == 'reflectance':
        idx = 2
    else:
        raise ValueError(f'Invalid img_type: {img_type}')

    output_img = input_img[:, idx * img_width:(idx + 1) * img_width]

    return output_img
