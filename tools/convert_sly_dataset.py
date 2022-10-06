import os
import json
import logging
import argparse
from pathlib import Path
from functools import partial
from joblib import Parallel, delayed
from typing import Dict, Tuple, List

import cv2
import pandas as pd
from tqdm import tqdm
import supervisely_lib as sly

from tools.supervisely_utils import read_sly_project
from tools.utils import crop_image, extract_body_part, extract_temperature


os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d.%m.%Y %H:%M:%S',
    filename='logs/{:s}.log'.format(Path(__file__).stem),
    filemode='w',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_class_id(
        class_label: str,
) -> int:

    if class_label == 'Zone':
        class_id = 1
    elif class_label == 'Artifact':
        class_id = 2
    else:
        raise ValueError(f'Unknown class_name: {class_label}')
    return class_id


def get_object_map(
        objects: List[Dict],
) -> dict:

    object_map = {}
    for obj in objects:
        object_map[obj['key']] = obj['classTitle']

    return object_map


def process_single_video(
        group: Tuple[int, pd.DataFrame],
        img_type: str,
        box_extension: int,
        save_dir: str,
) -> pd.DataFrame:
    _, row = group
    study_name = row['study']
    series_name = row['series']
    video_path = row['video_path']

    # Create image and annotation directories
    img_dir = os.path.join(save_dir, study_name, series_name, 'img')
    ann_dir = os.path.join(save_dir, study_name, series_name, 'ann')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    # Read video, save images, and empty annotations
    df_study = pd.DataFrame(
        columns=[
            'Study',
            'Series',
            'Body part',
            'Image path',
            'Ann path',
            'Temperature ID',
            'Temperature',
            'Wavelength ID',
            'Wavelength',
            'Class label',
            'Class ID',
            'x1',
            'y1',
            'x2',
            'y2',
            'xc',
            'yc',
            'Box width',
            'Box height',
            'Box area',
            'Image width',
            'Image height',
        ]
    )
    video = cv2.VideoCapture(video_path)
    temperature_idx, temperature = extract_temperature(path=series_name)

    idx = 0
    while True:
        success, _img = video.read()
        if not success:
            break
        img = crop_image(
            input_img=_img,
            img_type=img_type
        )
        wavelength_idx = idx + 1
        wavelength = 500 + 5*idx
        stem = f'{idx+1:03d}_W={wavelength:d}_T{temperature_idx:d}={temperature:s}'
        img_path = os.path.join(img_dir, f'{stem}.png')
        img_path_rel = os.path.relpath(img_path, start=save_dir)
        cv2.imwrite(img_path, img)

        ann_path = os.path.join(ann_dir, f'{stem}.txt')
        ann_path_rel = os.path.relpath(img_path, start=save_dir)
        f = open(file=ann_path, mode='w')
        f.close()

        # Add information to the data frame
        body_part = extract_body_part(img_path_rel)
        df_study.at[idx, 'Study'] = study_name
        df_study.at[idx, 'Series'] = series_name
        df_study.at[idx, 'Body part'] = body_part
        df_study.at[idx, 'Image path'] = img_path_rel
        df_study.at[idx, 'Ann path'] = ann_path_rel
        df_study.at[idx, 'Temperature ID'] = temperature_idx
        df_study.at[idx, 'Temperature'] = temperature
        df_study.at[idx, 'Wavelength ID'] = wavelength_idx
        df_study.at[idx, 'Wavelength'] = wavelength
        df_study.at[idx, 'Image width'] = img.shape[1]
        df_study.at[idx, 'Image height'] = img.shape[0]

        idx += 1

    # Read json with metadata
    ann_path = row['ann_path']
    f = open(ann_path)
    ann_data = json.load(f)
    img_height = ann_data['size']['height']
    img_width = ann_data['size']['width'] // 3          # delimiter is a number of images in a stacked row

    # Extract object keys
    object_map = get_object_map(objects=ann_data['objects'])

    # Save annotations
    for frame in ann_data['frames']:
        frame_idx = frame['index']
        ann_path = os.path.join(ann_dir, f'{frame_idx + 1:03d}.txt')
        f = open(file=ann_path, mode='w')
        for _figure in frame['figures']:

            _figure['geometry']['points']['exterior'][0][0] -= box_extension    # x1 or left
            _figure['geometry']['points']['exterior'][0][1] -= box_extension    # y1 or top
            _figure['geometry']['points']['exterior'][1][0] += box_extension    # x2 or right
            _figure['geometry']['points']['exterior'][1][1] += box_extension    # y2 or bottom

            figure = sly.Rectangle.from_json(_figure['geometry'])
            object_key = _figure['objectKey']
            class_label = object_map[object_key]
            class_id = get_class_id(class_label)

            # Move box if it was placed on the central or right image
            if figure.left > img_width:
                figure = figure.translate(drow=img_width, dcol=0)

            box_info = '{:s} {:d} {:d} {:d} {:d} {:d} {:d} {:d} {:d} {:d} {:d}\n'.format(
                class_label,                # label of the class e.g. Zone
                class_id,                   # id number of the class e.g. 1
                figure.left,                # x1
                figure.top,                 # y1
                figure.right,               # x2
                figure.bottom,              # y2
                figure.center.col,          # x of a center point
                figure.center.row,          # y of a center point
                figure.width,               # box width
                figure.height,              # box height
                int(figure.area),           # area of the box
            )
            f.write(box_info)

            df_study.at[frame_idx, 'Class label'] = class_label
            df_study.at[frame_idx, 'Class ID'] = class_id
            df_study.at[frame_idx, 'x1'] = figure.left
            df_study.at[frame_idx, 'y1'] = figure.top
            df_study.at[frame_idx, 'x2'] = figure.right
            df_study.at[frame_idx, 'y2'] = figure.bottom
            df_study.at[frame_idx, 'xc'] = figure.center.col
            df_study.at[frame_idx, 'yc'] = figure.center.row
            df_study.at[frame_idx, 'Box width'] = figure.height
            df_study.at[frame_idx, 'Box height'] = figure.height
            df_study.at[frame_idx, 'Box area'] = int(figure.area)

        f.close()

    return df_study


def main(
        df: pd.DataFrame,
        box_extension: int,
        img_type: str,
        save_dir: str,
) -> None:

    logger.info(f'Output directory.....: {save_dir}')
    logger.info(f'Box extension........: {box_extension}')

    study_list = list(set(df['study']))
    logger.info(f'Number of studies....: {len(study_list)}')

    processing_func = partial(
        process_single_video,
        img_type=img_type,
        box_extension=box_extension,
        save_dir=save_dir,
    )
    result = Parallel(n_jobs=-1)(
        delayed(processing_func)(row) for row in tqdm(df.iterrows(), desc='Dataset conversion', unit=' video')
    )
    df_out = pd.concat(result)
    df_out.reset_index(drop=True, inplace=True)
    df_out.sort_values(['Study', 'Series'], inplace=True)
    save_path = os.path.join(save_dir, f'metadata.csv')
    df_out.to_csv(
        save_path,
        index=False,
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Dataset conversion')
    parser.add_argument('--project_dir', required=True, type=str)
    parser.add_argument('--include_dirs', nargs='+', default=None, type=str)
    parser.add_argument('--exclude_dirs', nargs='+', default=None, type=str)
    parser.add_argument('--box_extension', default=0, type=int)
    parser.add_argument('--img_type',  default='absorbance', type=str, choices=['absorbance', 'hsv', 'reflectance'])
    parser.add_argument('--save_dir', required=True, type=str)
    args = parser.parse_args()

    df = read_sly_project(
        project_dir=args.project_dir,
        include_dirs=args.include_dirs,
        exclude_dirs=args.exclude_dirs,
    )

    main(
        df=df,
        box_extension=args.box_extension,
        img_type=args.img_type,
        save_dir=args.save_dir,
    )

    logger.info('')
    logger.info('Dataset conversion complete')
