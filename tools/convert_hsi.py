import os
import logging
import argparse
import multiprocessing
from pathlib import Path
from typing import List, Dict, Any
from functools import partial

import cv2
import numpy as np
import pandas as pd
from numpy import ndarray
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from tools.utils import get_file_list, read_hsi, extract_body_part, extract_time_stamp

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d.%m.%Y %I:%M:%S',
    filename='logs/{:s}.log'.format(Path(__file__).stem),
    filemode='w',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def process_hsi(
        file_path: str,
        save_dir: str,
) -> List:

    hsi = read_hsi(file_path)
    body_part = extract_body_part(file_path)
    date, time = extract_time_stamp(filename=Path(file_path).name)

    metadata = []
    for idx in range(hsi.shape[2]):
        image = hsi[:, :, idx]
        image_name = Path(file_path).stem + f'_{idx+1:03d}.png'
        image_path = os.path.join(save_dir, image_name)

        metadata.append(
            {
                'Source dir': str(Path(file_path).parent),
                'HSI name': Path(file_path).name,
                'Date': date,
                'Time': time,
                'Body part': body_part,
                'Image path': image_path,
                'Image name': image_name,
                'Min': np.min(image),
                'Mean': np.mean(image),
                'Max': np.max(image),
                'Height': image.shape[0],
                'Width': image.shape[1],
                'Wavelength': idx+1,
            }
        )
    logger.info(f'HSI processed......: {Path(file_path).name}')

    # TODO: convert hsi to images and save them

    return metadata


def main(
        input_dir: str,
        save_dir: str,
):
    # Log main parameters
    logger.info(f'Input dir..........: {input_dir}')
    logger.info(f'Output dir.........: {save_dir}')

    # Get list of HSI files
    hsi_list = get_file_list(
        src_dirs=input_dir,
        include_template='',
        ext_list='.dat',
    )
    logger.info(f'HSI files found....: {len(hsi_list)}')

    # Multiprocessing of HSI files
    os.makedirs(save_dir, exist_ok=True)
    num_cores = multiprocessing.cpu_count()
    processing_func = partial(
        process_hsi,
        save_dir=save_dir,
    )
    _metadata = process_map(
        processing_func,
        tqdm(hsi_list, desc='Process hyperspectral images', unit=' HSIs'),
        max_workers=num_cores,
    )
    metadata = sum(_metadata, [])

    # Save metadata as an xlsx file
    df = pd.DataFrame(metadata)
    df.sort_values('Image path', inplace=True)
    save_path = os.path.join(save_dir, 'metadata.xlsx')
    df.to_excel(
        save_path,
        sheet_name='Metadata',
        index=True,
        index_label='ID',
    )

    logger.info('')
    logger.info(f'Complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data processing')
    parser.add_argument('--input_dir', default='dataset', type=str)
    parser.add_argument('--include_dirs', nargs='+', default=None, type=str)
    parser.add_argument('--exclude_dirs', nargs='+', default=None, type=str)
    parser.add_argument('--save_dir', default='experiments', type=str)
    args = parser.parse_args()

    main(
        input_dir=args.input_dir,
        save_dir=args.save_dir,
    )

