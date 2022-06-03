import os
import logging
import argparse
import multiprocessing
from pathlib import Path
from functools import partial
from typing import List, Tuple

import cv2
import imutils
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from tools.utils import get_file_list, read_hsi, extract_body_part, extract_time_stamp, get_color_map

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
        data_type: str,
        color_map: str,
        apply_equalization: bool,
        output_type: str,
        output_size: Tuple[int, int],
        fps: int,
        save_dir: str,
) -> List:

    # Read HSI file and extract additional information
    hsi = read_hsi(
        path=file_path,
        data_type=data_type,
    )
    body_part = extract_body_part(file_path)
    date, time = extract_time_stamp(filename=Path(file_path).name)

    # Select save_dir based on output_type
    hsi_name = Path(file_path).stem
    if output_type == 'image':
        save_dir = os.path.join(save_dir, hsi_name)
    os.makedirs(save_dir, exist_ok=True)

    # Create video writer
    if output_type == 'video':
        video_name = f'{hsi_name}.mp4'
        video_path = os.path.join(save_dir, video_name)
        _img = imutils.resize(hsi[:, :, 0], height=output_size[0], inter=cv2.INTER_LINEAR)
        _img_size = _img.shape[:-1] if len(_img.shape) == 3 else _img.shape
        video_height, video_width = _img_size
        video = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (video_width, video_height),
        )

    # Process HSI in an image-by-image fashion
    metadata = []
    for idx in range(hsi.shape[2]):

        img = hsi[:, :, idx]
        img_name = Path(file_path).stem + f'_{idx+1:03d}.png'
        img_path = os.path.join(save_dir, img_name)

        metadata.append(
            {
                'Source dir': str(Path(file_path).parent),
                'HSI name': str(Path(file_path).name),
                'Date': date,
                'Time': time,
                'Body part': body_part,
                'Image path': img_path,
                'Image name': img_name,
                'Min': np.min(img),
                'Mean': np.mean(img),
                'Max': np.max(img),
                'Height': img.shape[0],
                'Width': img.shape[1],
                'Wavelength': idx+1,
            }
        )

        # Resize and normalize image
        img_size = img.shape[:-1] if len(img.shape) == 3 else img.shape
        if img_size != output_size:
            img = imutils.resize(img, height=output_size[0], inter=cv2.INTER_LINEAR)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Equalize histogram
        if apply_equalization:
            img = cv2.equalizeHist(img)

        # Colorize image
        if isinstance(color_map, str) and color_map is not None:
            cmap = get_color_map(color_map)
            img = cv2.applyColorMap(img, cmap)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Save image
        if output_type == 'image':
            cv2.imwrite(img_path, img)
        elif output_type == 'video':
            video.write(img)
        else:
            raise ValueError(f'Unknown output_type value: {output_type}')

    video.release() if output_type == 'video' else False

    logger.info(f'HSI processed......: {Path(file_path).name}')

    return metadata


def main(
        input_dir: str,
        data_type: str,
        color_map: str,
        apply_equalization: bool,
        output_type: str,
        output_size: Tuple[int, int],
        fps: int,
        save_dir: str,
):

    # Log main parameters
    logger.info(f'Input dir..........: {input_dir}')
    logger.info(f'Output dir.........: {save_dir}')
    logger.info(f'Data type..........: {data_type}')
    logger.info(f'Color map..........: {color_map}')
    logger.info(f'Apply equalization.: {apply_equalization}')
    logger.info(f'Output type........: {output_type}')
    logger.info(f'Output size........: {output_size}')
    logger.info(f'FPS................: {fps if output_type == "video" else None}')
    logger.info('')

    # Get list of HSI files
    hsi_paths = get_file_list(
        src_dirs=input_dir,
        include_template='',
        ext_list='.dat',
    )
    logger.info(f'HSI found..........: {len(hsi_paths)}')

    # Multiprocessing of HSI files
    os.makedirs(save_dir, exist_ok=True)
    num_cores = multiprocessing.cpu_count()
    processing_func = partial(
        process_hsi,
        data_type=data_type,
        color_map=color_map,
        apply_equalization=apply_equalization,
        output_type=output_type,
        output_size=output_size,
        fps=fps,
        save_dir=save_dir,
    )
    _metadata = process_map(
        processing_func,
        tqdm(hsi_paths, desc='Process hyperspectral images', unit=' HSI'),
        max_workers=num_cores,
    )
    metadata = sum(_metadata, [])

    # Save metadata as an XLSX file
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

    parser = argparse.ArgumentParser(description='Convert Hyperspectral Images')
    parser.add_argument('--input_dir', default='dataset', type=str)
    parser.add_argument('--data_type',  default='absorbance', type=str, choices=['absorbance', 'reflectance'])
    parser.add_argument('--color_map', default=None, type=str, choices=['jet', 'bone', 'ocean', 'cool', 'hsv'])
    parser.add_argument('--apply_equalization', action='store_true')
    parser.add_argument('--output_type', default='image', type=str, choices=['image', 'video'])
    parser.add_argument('--output_size', default=[744, 1000], nargs='+', type=int)
    parser.add_argument('--fps', default=15, type=int)
    parser.add_argument('--save_dir', default='dataset/converted', type=str)
    args = parser.parse_args()

    main(
        input_dir=args.input_dir,
        data_type=args.data_type,
        color_map=args.color_map,
        apply_equalization=args.apply_equalization,
        output_type=args.output_type,
        output_size=tuple(args.output_size),
        fps=args.fps,
        save_dir=args.save_dir,
    )
