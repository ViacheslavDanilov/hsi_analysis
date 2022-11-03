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
        modality: str = 'absorbance',
) -> np.ndarray:
    """
    Read HSI images (.dat files) as a numpy array (H, W, WL)

    Args:
        path: a path to a .dat file being read
        modality: a mode of reading HSI images
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

        if modality == 'reflectance':
            pass
        elif modality == 'absorbance':
            data = -np.log10(np.maximum(data, 0.01))
        else:
            raise ValueError(f'Invalid data type: {modality}')

        data = data.astype(np.float32)

        return data


def resize_volume(
        input_image: np.ndarray,
        output_size: Tuple[int, int, int],
        interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:

    output_height, output_width, output_depth = output_size
    source_height, source_width, source_depth = input_image.shape

    intermediate_image = np.zeros(
        shape=(output_height, output_width, source_depth),
        dtype=input_image.dtype,
    )
    output_image = np.zeros(
        shape=(output_height, output_width, output_depth),
        dtype=input_image.dtype,
    )

    # Resize along height and width dimensions
    if (output_height, output_width) != (source_height, source_width):
        for i in range(source_depth):
            img = input_image[:, :, i]
            img_out = cv2.resize(
                src=img,
                dsize=(output_width, output_height),
                interpolation=interpolation,
            )
            intermediate_image[:, :, i] = img_out
    else:
        intermediate_image = input_image.copy()

    # Resize along width and depth dimensions
    if (output_width, output_depth) != (source_width, source_depth):
        for j in range(intermediate_image.shape[0]):
            img = intermediate_image[j, :, :]
            img_out = cv2.resize(
                src=img,
                dsize=(output_depth, output_width),
                interpolation=interpolation,
            )
            output_image[j, :, :] = img_out
    else:
        output_image = intermediate_image.copy()

    return output_image


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


def get_study_name(
        path: str,
) -> str:
    study_name = Path(path).parts[-3]
    return study_name


def get_series_name(
        path: str,
) -> str:
    series_name = Path(path).parts[-2]
    return series_name


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


def extract_temperature(
        path: str,
) -> Tuple[int, str]:

    if os.path.isfile(path):
        series_name = get_series_name(path)
    else:
        series_name = path

    _temperature_idx = re.findall(r'\d+', series_name)
    temperature_idx = int(_temperature_idx[3])

    idx = series_name.index('=') + 1
    temperature = series_name[idx:]

    return temperature_idx, temperature


def extract_time_stamp(
        path: str,
) -> Tuple[str, str]:
    """
    Extract a time stamp from the HSI path

    Args:
        path: a path to filename with an unstructured time stamp
    Returns:
        date: a date string in format DD.MM.YYYY
        time: a time string in format HH:MM:SS
    """

    study_name = get_study_name(path)
    series_name = get_series_name(path)

    try:
        date_list = re.findall(r'\d+', study_name)
        time_list = re.findall(r'\d+', series_name)
        date = f'{date_list[0]}.{date_list[1]}.{date_list[2]}'
        time = f'{time_list[0]}:{time_list[1]}:{time_list[2]}'
    except Exception as e:
        date = 'Invalid_date'
        time = 'Invalid_time'

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
    assert input_img.shape[1] % 3 == 0, 'Input image width should be divisible by 3 (contain 3 sub-images)'

    img_width = int(input_img.shape[1] / 3)
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
