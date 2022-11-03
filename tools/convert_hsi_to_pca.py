import os
import logging
import argparse
import multiprocessing
from pathlib import Path
from functools import partial
from typing import List, Union, Tuple, Optional

import cv2
import ffmpeg
import imutils
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from tools.utils import (
    read_hsi,
    get_file_list,
    get_dir_list,
    get_study_name,
    get_series_name,
    get_color_map,
    extract_body_part,
    extract_temperature,
    extract_time_stamp,
)

os.makedirs('../logs', exist_ok=True)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d.%m.%Y %H:%M:%S',
    filename='logs/{:s}.log'.format(Path(__file__).stem),
    filemode='w',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def process_hsi(
        hsi_path: str,
        save_dir: str,
        num_components: int = 10,
        modality: str = 'absorbance',
        color_map: Optional[str] = None,
        apply_equalization: bool = False,
        output_size: Tuple[int, int] = (744, 1000),
) -> List:

    # Read HSI file and extract additional information
    hsi = read_hsi(
        path=hsi_path,
        modality=modality,
    )
    study_name = get_study_name(path=hsi_path)
    series_name = get_series_name(path=hsi_path)
    body_part = extract_body_part(path=hsi_path)
    temperature_idx, temperature = extract_temperature(path=hsi_path)
    date, time = extract_time_stamp(path=hsi_path)

    metadata = []

    # TODO: add PCA processing

    return metadata


def main(
        input_dir: str,
        save_dir: str,
        num_components: int = 10,
        modality: str = 'absorbance',
        color_map: Optional[str] = None,
        apply_equalization: bool = False,
        output_size: Tuple[int, int] = (744, 1000),
        include_dirs: Optional[Union[List[str], str]] = None,
        exclude_dirs: Optional[Union[List[str], str]] = None,
) -> None:

    # Log main parameters
    logger.info(f'Input dir..........: {input_dir}')
    logger.info(f'Included dirs......: {include_dirs}')
    logger.info(f'Excluded dirs......: {exclude_dirs}')
    logger.info(f'Components.........: {num_components}')
    logger.info(f'Modality...........: {modality}')
    logger.info(f'Color map..........: {color_map}')
    logger.info(f'Apply equalization.: {apply_equalization}')
    logger.info(f'Output size........: {output_size}')
    logger.info(f'Output dir.........: {save_dir}')
    logger.info('')

    # Filter the list of studied directories
    study_dirs = get_dir_list(
        data_dir=input_dir,
        include_dirs=include_dirs,
        exclude_dirs=exclude_dirs,
    )

    # Get list of HSI files
    hsi_paths = get_file_list(
        src_dirs=study_dirs,
        include_template='',
        ext_list='.dat',
    )
    logger.info(f'HSI found..........: {len(hsi_paths)}')

    # Multiprocessing of HSI files
    os.makedirs(save_dir, exist_ok=True)
    processing_func = partial(
        process_hsi,
        modality=modality,
        color_map=color_map,
        apply_equalization=apply_equalization,
        output_size=output_size,
        save_dir=save_dir,
    )
    # num_cores = multiprocessing.cpu_count()         # TODO: uncomment
    num_cores = 1
    res = process_map(
        processing_func,
        tqdm(hsi_paths, desc='Process hyperspectral images', unit=' HSI'),
        max_workers=num_cores,
    )
    metadata = sum(res, [])

    # Save metadata as an XLSX file
    df = pd.DataFrame(metadata)
    df.sort_values('Source dir', inplace=True)
    df.sort_index(inplace=True)
    save_path = os.path.join(save_dir, 'pca.xlsx')
    df.index += 1
    df.to_excel(
        save_path,
        sheet_name='PCA',
        index=True,
        index_label='ID',
    )
    logger.info('')
    logger.info(f'Complete')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Reduce HSI cubes')
    parser.add_argument('--input_dir', default='dataset/HSI', type=str)
    parser.add_argument('--include_dirs', nargs='+', default=None, type=str)
    parser.add_argument('--exclude_dirs', nargs='+', default=None, type=str)
    parser.add_argument('--num_components', default=10, type=int)
    parser.add_argument('--modality',  default='absorbance', type=str, choices=['absorbance', 'reflectance'])
    parser.add_argument('--color_map', default=None, type=str, choices=['jet', 'bone', 'ocean', 'cool', 'hsv'])
    parser.add_argument('--apply_equalization', action='store_true')
    parser.add_argument('--output_size', default=[744, 1000], nargs='+', type=int)
    parser.add_argument('--save_dir', default='dataset/HSI_pca', type=str)
    args = parser.parse_args()

    main(
        input_dir=args.input_dir,
        include_dirs=args.include_dirs,
        exclude_dirs=args.exclude_dirs,
        modality=args.modality,
        num_components=args.num_components,
        color_map=args.color_map,
        apply_equalization=args.apply_equalization,
        output_size=tuple(args.output_size),
        save_dir=args.save_dir,
    )
