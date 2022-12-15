import os
import json
import logging
import argparse
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from tools.supervisely_utils import read_sly_project
from settings import (
    SLY_PROJECT_DIR,
    INCLUDE_DIRS,
    EXCLUDE_DIRS,
    TRAIN_SIZE,
    BOX_EXTENSION,
    SEED,
    COCO_SAVE_DIR,
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


def add_metadata(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df['PC'] = df['series'].apply(lambda x: int(x.split('_')[-1]))
    df['series'] = df['series'].apply(lambda x: '_'.join(x.split('_')[:-1]))

    return df


def prepare_subsets(
    df: pd.DataFrame,
    train_size: float = 0.80,
    seed: int = 11,
) -> dict:

    subsets = {
        'train': {'images': [], 'labels': []},
        'test': {'images': [], 'labels': []},
    }

    series_list = list(set(df['series']))
    series_train, series_test = train_test_split(
        series_list,
        train_size=train_size,
        shuffle=True,
        random_state=seed,
    )
    df_train = df[df['series'].isin(series_train)]
    df_test = df[df['series'].isin(series_test)]

    subsets['train']['images'].extend(df_train['img_path'])
    subsets['train']['labels'].extend(df_train['ann_path'])
    subsets['test']['images'].extend(df_test['img_path'])
    subsets['test']['labels'].extend(df_test['ann_path'])

    logger.info('')
    logger.info('Overall train/test split')
    logger.info(f'Studies...................: {df_train["study"].nunique()}/{df_test["study"].nunique()}')
    logger.info(f'Series....................: {df_train["series"].nunique()}/{df_test["series"].nunique()}')
    logger.info(f'Images....................: {len(df_train)}/{len(df_test)}')

    assert len(subsets['train']['images']) == len(subsets['train']['labels']), 'Mismatch length of the training subset'
    assert len(subsets['test']['images']) == len(subsets['test']['labels']), 'Mismatch length of the testing subset'

    return subsets


def prepare_coco(
    subsets: dict,
    box_extension: dict,
    save_dir: str,
) -> None:

    # TODO: add support for COCO dataset parsing
    print('Complete')


def main(
    df: pd.DataFrame,
    box_extension: dict,
    save_dir: str,
    train_size: float = 0.80,
    seed: int = 0,
) -> None:

    logger.info(f'Output directory.....: {save_dir}')
    logger.info(f'Box extension........: {box_extension}')
    logger.info(f'Number of studies....: {len(set(df["study"]))}')

    df = add_metadata(df)

    subsets = prepare_subsets(
        df=df,
        train_size=train_size,
        seed=seed,
    )

    prepare_coco(
        subsets=subsets,
        box_extension=box_extension,
        save_dir=save_dir,
    )

    logger.info('Complete')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Dataset conversion')
    parser.add_argument('--project_dir', default=SLY_PROJECT_DIR, type=str)
    parser.add_argument('--include_dirs', nargs='+', default=INCLUDE_DIRS, type=str)
    parser.add_argument('--exclude_dirs', nargs='+', default=EXCLUDE_DIRS, type=str)
    parser.add_argument('--train_size', default=TRAIN_SIZE, type=float)
    parser.add_argument('--box_extension', default=BOX_EXTENSION)
    parser.add_argument('--seed', default=SEED, type=int)
    parser.add_argument('--save_dir', default=COCO_SAVE_DIR, type=str)
    args = parser.parse_args()

    df = read_sly_project(
        project_dir=args.project_dir,
        include_dirs=args.include_dirs,
        exclude_dirs=args.exclude_dirs,
    )

    main(
        df=df,
        train_size=args.train_size,
        box_extension=args.box_extension,
        seed=args.seed,
        save_dir=args.save_dir,
    )
