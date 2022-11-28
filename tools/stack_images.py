import os
import logging
import argparse
from pathlib import Path
from functools import partial
from joblib import Parallel, delayed
from typing import List, Tuple, Union, Optional

import cv2
import ffmpeg
import imutils
import numpy as np
import pandas as pd
from tqdm import tqdm

from tools.utils import (
    get_file_list,
    get_dir_list,
    extract_study_name,
    extract_series_name,
)

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d.%m.%Y %H:%M:%S',
    filename='logs/{:s}.log'.format(Path(__file__).stem),
    filemode='w',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def stack_images(
        img_paths: Tuple[str, ...],
        output_size: Tuple[int, int] = (744, 1000),
) -> np.ndarray:

    # Read images and stack them together
    img_out = np.zeros([output_size[0], 1, 3], dtype=np.uint8)
    for img_path in img_paths:
        img = cv2.imread(img_path)
        img_size = img.shape[:-1] if len(img.shape) == 3 else img.shape
        if img_size != output_size:
            img = imutils.resize(img, height=output_size[0], inter=cv2.INTER_LINEAR)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img_out = np.hstack([img_out, img])

    # Delete 1 pixel-width column added at the beginning
    img_out = np.delete(img_out, 0, 1)

    return img_out


def process_group(
        series_group: tuple,
        save_dir: str,
        output_type: str = 'image',
        output_size: Tuple[int, int] = (744, 1000),
        fps: int = 15,
) -> None:

    (study_name, series_name), df = series_group

    if output_type == 'image':
        output_dir = os.path.join(save_dir, study_name, series_name)
    else:
        output_dir = os.path.join(save_dir, study_name)
    os.makedirs(output_dir, exist_ok=True)

    if output_type == 'video':
        video_path_temp = os.path.join(output_dir, f'{series_name}_temp.mp4')
        num_studies = len(df['dir_id'].unique())
        video_height, video_width = output_size[0], num_studies*output_size[1]
        video = cv2.VideoWriter(
            video_path_temp,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (video_width, video_height),
        )

    img_groups = df.groupby('img_name')

    for idx, (img_name, df_img) in enumerate(img_groups):

        assert idx + 1 == int(Path(img_name).stem), 'Wavelength and image name mismatch'

        # Stack images
        img_paths = df_img['img_path'].tolist()
        img = stack_images(img_paths, output_size)

        # Save image
        if output_type == 'image':
            save_path = os.path.join(output_dir, img_name)
            cv2.imwrite(save_path, img)
        elif output_type == 'video':
            video.write(img)
        else:
            raise ValueError(f'Unknown output_type value: {output_type}')

    video.release() if output_type == 'video' else False

    # Replace OpenCV videos with FFmpeg ones
    if output_type == 'video':
        video_path = os.path.join(output_dir, f'{series_name}.mp4')
        stream = ffmpeg.input(video_path_temp)
        stream = ffmpeg.output(stream, video_path, vcodec='libx264', video_bitrate='10M')
        ffmpeg.run(stream, quiet=True, overwrite_output=True)
        os.remove(video_path_temp)


def main(
        input_dirs: List[str],
        save_dir: str,
        output_type: str = 'image',
        output_size: Tuple[int, int] = (744, 1000),
        fps: int = 15,
        include_dirs: Optional[Union[List[str], str]] = None,
        exclude_dirs: Optional[Union[List[str], str]] = None,
) -> None:
    """
    Stack images from multiple directories into one image

    Args:
        input_dirs: list of dirs with source images
        save_dir: directory where arranged images are saved
        output_type: whether to save it as a video or image
        output_size: new size of images (the algorithm keeps the aspect ratio)
        fps: frames per second of the output video
        include_dirs: directories to be included
        exclude_dirs: directories to be excluded
    Returns: None
    """

    # Collect images across all dirs
    img_dirs = []
    for input_dir in input_dirs:

        study_dirs = get_dir_list(
            data_dir=input_dir,
            include_dirs=include_dirs,
            exclude_dirs=exclude_dirs,
        )

        _img_paths = get_file_list(
            src_dirs=study_dirs,
            include_template='',
            ext_list='.png',
        )
        img_dirs.append(_img_paths)

    # Check if all dirs have the same number of images
    num_images = [len(img_dir) for img_dir in img_dirs]
    assert len(set(num_images)) == 1, 'Mismatch in number of images across input directories'

    # Create a dataframe with all images
    df = pd.DataFrame()
    for idx, img_dir in enumerate(img_dirs):
        _study_list = map(lambda path: extract_study_name(path), img_dir)
        _series_list = map(lambda path: extract_series_name(path), img_dir)
        _img_names = map(lambda path: os.path.basename(path), img_dir)
        _df = pd.DataFrame(img_dir, columns=['img_path'])
        _df['img_name'] = list(_img_names)
        _df['study'] = list(_study_list)
        _df['series'] = list(_series_list)
        _df['dir_id'] = idx + 1
        df = pd.concat([df, _df], ignore_index=True)

    # Group images and process them
    gb = df.groupby(['study', 'series'])
    processing_func = partial(
        process_group,
        output_type=output_type,
        output_size=output_size,
        fps=fps,
        save_dir=save_dir,
    )
    Parallel(n_jobs=-1, prefer='threads')(
        delayed(processing_func)(group) for group in tqdm(gb, desc='Stacking images', unit=' groups')
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Stack images')
    parser.add_argument('--input_dirs', nargs='+', required=True, type=str)
    parser.add_argument('--include_dirs', nargs='+', default=None, type=str)
    parser.add_argument('--exclude_dirs', nargs='+', default=None, type=str)
    parser.add_argument('--output_type', default='image', type=str, choices=['image', 'video'])
    parser.add_argument('--output_size', default=[744, 1000], nargs='+', type=int)
    parser.add_argument('--fps', default=15, type=int)
    parser.add_argument('--save_dir', default='dataset/HSI_processed/stack', type=str)
    args = parser.parse_args()

    main(
        input_dirs=args.input_dirs,
        include_dirs=args.include_dirs,
        exclude_dirs=args.exclude_dirs,
        output_type=args.output_type,
        output_size=args.output_size,
        fps=args.fps,
        save_dir=args.save_dir,
    )
