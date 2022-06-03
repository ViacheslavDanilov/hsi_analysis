import os
import logging
import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import imutils
import numpy as np
import pandas as pd

from tools.utils import get_file_list

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d.%m.%Y %I:%M:%S',
    filename='logs/{:s}.log'.format(Path(__file__).stem),
    filemode='w',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main(
        input_dirs: List[str],
        output_type: str,
        output_size: Tuple[int, int],
        fps: int,
        save_dir: str,
) -> None:
    """
    Stack images from multiple directories into one image.

    Args:
        input_dirs: list of dirs with source images
        output_type: whether to save it as a video or image
        output_size: new size of images (the algorithm keeps the aspect ratio)
        fps: frames per second of the output video
        save_dir: directory where arranged images are saved
    Returns: None
    """

    # Collect images across all dirs
    img_dirs = []
    for input_dir in input_dirs:
        _img_paths = get_file_list(
            src_dirs=input_dir,
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
        _dir_list = map(lambda path: path.split(os.sep)[2], img_dir)
        _img_names = map(lambda path: os.path.basename(path), img_dir)
        _df = pd.DataFrame(img_dir, columns=['image_path'])
        _df['image_name'] = list(_img_names)
        _df['dir_name'] = list(_dir_list)
        _df['dir_id'] = idx + 1
        df = pd.concat([df, _df], ignore_index=True)

    # Group images by their name and then stack them
    gb = df.groupby('image_name')

    # TODO: save images as a video of a set of images
    for idx, (img_name, sample) in enumerate(gb):
        img_paths = sample['image_path'].tolist()
        img = stack_images(img_paths, output_size)


def stack_images(
        img_paths: Tuple[str, ...],
        output_size: Tuple[int, int],
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arrange annotations')
    parser.add_argument('--input_dirs', nargs='+', required=True, type=str)
    parser.add_argument('--output_type', default='image', type=str, choices=['image', 'video'])
    parser.add_argument('--output_size', default=[744, 1000], nargs='+', type=int)
    parser.add_argument('--fps', default=15, type=int)
    parser.add_argument('--save_dir', required=True, type=str)
    args = parser.parse_args()

    main(
        input_dirs=args.input_dirs,
        output_type=args.output_type,
        output_size=args.output_size,
        fps=args.fps,
        save_dir=args.save_dir,
    )
