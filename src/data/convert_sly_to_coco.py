import logging
import multiprocessing
import os
from functools import partial
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from src.data.utils import extract_body_part, extract_study_name, extract_temperature, get_file_list

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def filter_sly_dirs(
    src_dir: str,
    abs: bool = True,
    ref: bool = False,
    pca: bool = False,
    tsne: bool = True,
) -> List[str]:

    dir_list = glob(src_dir + '/*/*/', recursive=True)

    # Filter reduction directories
    initial_list = []
    if pca:
        red_list_pca = list(filter(lambda path: 'pca' in path, dir_list))
        initial_list.extend(red_list_pca)

    if tsne:
        red_list_tsne = list(filter(lambda path: 'tsne' in path, dir_list))
        initial_list.extend(red_list_tsne)

    # Filter modality directories
    output_list = []
    if abs:
        mod_list_abs = list(filter(lambda path: 'abs' in path, initial_list))
        output_list.extend(mod_list_abs)

    if ref:
        mod_list_ref = list(filter(lambda path: 'ref' in path, initial_list))
        output_list.extend(mod_list_ref)

    return output_list


def add_metadata(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Enrich metadata of the dataset.

    Args:
        df: a source data frame with image and annotation paths
    Returns:
        df: an updated data frame with split series into series names and PCs
    """
    df['stem'] = df['img_path'].apply(lambda x: Path(x).stem)
    df['component'] = df['stem'].apply(lambda x: int(x.split('_')[-1]))
    df['body_part'] = df['img_path'].apply(extract_body_part)
    df['temperature'] = df['stem'].apply(lambda x: extract_temperature)

    for idx, row in df.iterrows():
        temperature_idx, temperature_ = extract_temperature(row['stem'])
        temperature = temperature_.split('_')[0]
        df.at[idx, 'temperature_idx'] = temperature_idx
        df.at[idx, 'temperature'] = temperature

    for idx, row in df.iterrows():
        symbol_idx = row['stem'].rfind('_')
        series_name = row['stem'][:symbol_idx]
        df.at[idx, 'series'] = series_name
    df['study'] = df['img_path'].apply(extract_study_name)

    return df


def split_dataset(
    df: pd.DataFrame,
    train_size: float = 0.80,
    seed: int = 11,
) -> dict:
    """Split dataset into training and test subsets.

    Args:
        df: data frame derived from a metadata file
        train_size: a fraction used to split dataset into train and test subsets
        seed: random value for splitting train and test subsets
    Returns:
        subsets: dictionary which contains image/annotation paths for train and test subsets
    """
    subsets: Dict[str, Dict[str, List[str]]] = {
        'train': {'images': [], 'labels': []},
        'test': {'images': [], 'labels': []},
    }

    df_temp = df[['body_part', 'series']]
    df_temp.drop_duplicates(inplace=True)
    series_train, series_test = train_test_split(
        df_temp['series'],
        train_size=train_size,
        shuffle=True,
        random_state=seed,
        stratify=df_temp['body_part'],
    )

    df_train = df[df['series'].isin(series_train)]
    df_test = df[df['series'].isin(series_test)]

    subsets['train']['images'].extend(df_train['img_path'])
    subsets['train']['labels'].extend(df_train['ann_path'])
    subsets['test']['images'].extend(df_test['img_path'])
    subsets['test']['labels'].extend(df_test['ann_path'])
    assert len(subsets['train']['images']) == len(
        subsets['train']['labels'],
    ), 'Mismatch length of the training subset'
    assert len(subsets['test']['images']) == len(
        subsets['test']['labels'],
    ), 'Mismatch length of the testing subset'

    log.info('')
    log.info(f'Split............: Train / Test')
    log.info(f'Studies..........: {df_train["study"].nunique()}/{df_test["study"].nunique()}')
    log.info(f'Series...........: {df_train["series"].nunique()}/{df_test["series"].nunique()}')
    log.info(f'Images...........: {len(df_train)}/{len(df_test)}')

    df_train_dist = pd.DataFrame()
    df_train_dist['Abs'] = df_train['body_part'].value_counts(normalize=False)
    df_train_dist['Rel'] = df_train['body_part'].value_counts(normalize=True)

    df_test_dist = pd.DataFrame()
    df_test_dist['Abs'] = df_test['body_part'].value_counts(normalize=False)
    df_test_dist['Rel'] = df_test['body_part'].value_counts(normalize=True)

    for idx in range(len(df_train_dist)):
        body_part = df_train_dist.index[idx]
        log.info(f'')
        log.info(f'Body part........: {body_part}')
        log.info(
            f'Absolute.........: {df_train_dist.loc[body_part, "Abs"]}/{df_test_dist.loc[body_part, "Abs"]}',
        )
        log.info(
            f'Relative.........: {df_train_dist.loc[body_part, "Rel"]:.2f}/{df_test_dist.loc[body_part, "Rel"]:.2f}',
        )

    return subsets


# TODO: Verify if this is still needed
def convert_single_study(
    study_dir: str,
    output_type: str,
    output_size: Tuple[int, int],
    to_gray: bool,
    fps: int,
    save_dir: str,
) -> None:
    print('')


@hydra.main(config_path=os.path.join(os.getcwd(), 'config'), config_name='data', version_base=None)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Filter used directories
    sly_dirs = filter_sly_dirs(
        src_dir=cfg.conversion.src_dir,
        abs=cfg.conversion.abs,
        ref=cfg.conversion.ref,
        pca=cfg.conversion.pca,
        tsne=cfg.conversion.tsne,
    )

    # Get list of images and annotations
    img_list = []
    ann_list = []
    for sly_dir in sly_dirs:
        img_list_ = get_file_list(
            src_dirs=sly_dir,
            ext_list='.png',
        )
        ann_list_ = get_file_list(
            src_dirs=sly_dir,
            ext_list='.json',
        )
        img_list.extend(img_list_)
        ann_list.extend(ann_list_)

    # Add additional metadata
    ann_list = [path for path in ann_list if Path(path).name != 'meta.json']
    df = pd.DataFrame(
        {
            'img_path': img_list,
            'ann_path': ann_list,
        },
    )
    df = add_metadata(df)

    # Split dataset using body part stratification
    subsets = split_dataset(
        df,
        train_size=cfg.conversion.train_size,
        seed=cfg.conversion.seed,
    )

    # TODO: prepare COCO dataset

    # Process data in parallel
    num_cores = multiprocessing.cpu_count()
    conversion_func = partial(
        convert_single_study,
        output_type=cfg.conversion.output_type,
        output_size=cfg.conversion.output_size,
        to_gray=cfg.conversion.to_gray,
        fps=cfg.conversion.fps,
        save_dir=cfg.conversion.save_dir,
    )
    process_map(
        conversion_func,
        tqdm(ann_list, desc='Convert studies', unit=' study'),
        max_workers=num_cores,
    )
    log.info('Complete')


if __name__ == '__main__':
    main()
