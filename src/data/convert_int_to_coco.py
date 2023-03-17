import logging
import os
from glob import glob
from typing import List

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def filter_dirs(
    src_dir: str,
    pca: bool = True,
    tsne: bool = False,
    abs: bool = True,
    ref: bool = False,
) -> List[str]:

    dir_list = glob(src_dir + '/*/', recursive=True)

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


def split_dataset(
    df: pd.DataFrame,
    train_size: float = 0.80,
    seed: int = 11,
) -> pd.DataFrame:
    """Split dataset into training and test subsets.

    Args:
        df: data frame derived from a metadata file
        train_size: a fraction used to split dataset into train and test subsets
        seed: random value for splitting train and test subsets
    Returns:
        subsets: dictionary which contains image/annotation paths for train and test subsets
    """
    df_temp = df[['body_part', 'series']]
    df_temp = df_temp.drop_duplicates()
    series_train, series_test = train_test_split(
        df_temp['series'],
        train_size=train_size,
        shuffle=True,
        random_state=seed,
        stratify=df_temp['body_part'],
    )

    df_train = df[df['series'].isin(series_train)]
    df_train = df_train.assign(split='train')
    df_test = df[df['series'].isin(series_test)]
    df_test = df_test.assign(split='test')
    df_out = pd.concat([df_train, df_test])
    df_out.reset_index(drop=True, inplace=True)

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

    return df_out


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'config'),
    config_name='convert_int_to_coco',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Filter used directories
    data_dirs = filter_dirs(
        src_dir=cfg.src_dir,
        abs=cfg.abs,
        ref=cfg.ref,
        pca=cfg.pca,
        tsne=cfg.tsne,
    )

    # Split datasets
    df_split = pd.DataFrame()
    for data_dir in data_dirs:
        meta_path = os.path.join(data_dir, 'metadata.xlsx')
        df = pd.read_excel(meta_path)
        df_split_ = split_dataset(
            df=df,
            train_size=cfg.train_size,
            seed=cfg.seed,
        )
        df_split = pd.concat([df_split, df_split_])
    df_split.drop('ID', axis=1, inplace=True)

    # TODO: add prepare_coco_subsets function
    # TODO: copy images to the coco dir
    # TODO: save metadata

    log.info('')
    log.info(f'Complete')


if __name__ == '__main__':
    main()
