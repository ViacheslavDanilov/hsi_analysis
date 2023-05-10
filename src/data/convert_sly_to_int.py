import logging
import os
import shutil
from functools import partial
from pathlib import Path

import hydra
import pandas as pd
import supervisely_lib as sly
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data.utils import extract_body_part, extract_study_name, extract_temperature

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def process_dataset(
    dataset: sly.Dataset,
    meta: sly.ProjectMeta,
) -> pd.DataFrame:
    df = pd.DataFrame(
        columns=[
            'img_path',
            'img_name',
            'img_height',
            'img_width',
            'stem',
            'series',
            'study',
            'reduction',
            'modality',
            'component',
            'body_part',
            'temperature',
            'temperature_idx',
            'source_type',
            'x1',
            'y1',
            'x2',
            'y2',
            'xc',
            'yc',
            'class',
            'box_width',
            'box_height',
            'area',
        ],
    )

    # Iterate over principal components
    for sample in dataset:
        img_path = dataset.get_img_path(sample)
        ann_path = dataset.get_ann_path(sample)
        img_name = Path(img_path).name
        ann_name = Path(ann_path).name

        stem = Path(img_name).stem
        series_parts = stem.split('_')[:-1]
        series_name = '_'.join(series_parts)
        study_name = extract_study_name(img_path, idx=-3)
        temperature_idx, temperature_ = extract_temperature(stem)
        temperature = temperature_.split('_')[0]

        ann = sly.Annotation.load_json_file(ann_path, meta)

        metadata = {
            'img_path': img_path,
            'img_name': img_name,
            'img_height': ann.img_size[0],
            'img_width': ann.img_size[1],
            'stem': stem,
            'series': series_name,
            'study': study_name,
            'reduction': Path(img_path).parts[-5],
            'modality': Path(img_path).parts[-4],
            'component': int(stem.split('_')[-1]),
            'body_part': extract_body_part(img_path),
            'temperature': temperature,
            'temperature_idx': temperature_idx,
        }

        for label in ann.labels:
            metadata['source_type'] = label.geometry.geometry_name()
            metadata['x1'] = label.geometry.left
            metadata['y1'] = label.geometry.top
            metadata['x2'] = label.geometry.right
            metadata['y2'] = label.geometry.bottom
            metadata['xc'] = label.geometry.center.col
            metadata['yc'] = label.geometry.center.row
            metadata['class'] = label.obj_class.name
            metadata['box_width'] = label.geometry.width
            metadata['box_height'] = label.geometry.height
            metadata['area'] = int(label.area)

        df = df.append(metadata, ignore_index=True)

    return df


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'config'),
    config_name='convert_sly_to_int',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    project = sly.Project(cfg.sly_dir, sly.OpenMode.READ)
    meta = project.meta

    # Extract metadata
    processing_func = partial(
        process_dataset,
        meta=meta,
    )
    result = Parallel(n_jobs=-1)(
        delayed(processing_func)(dataset)
        for dataset in tqdm(project, desc='Supervisely-to-Intermediate Conversion', unit=' series')
    )
    df = pd.concat(result)
    df.sort_values(['img_path', 'component'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.index += 1

    log.info(f'Studies............: {len(df["study"].unique())}')
    log.info(f'Series.............: {len(df["series"].unique())}')
    log.info(f'Images.............: {len(df["img_path"].unique())}')

    # Extract reduction technique and modality
    reduction_ = df['reduction'].unique()
    assert (
        reduction_.shape[0] == 1
    ), 'Multiple reduction techniques are mixed. There should be only one'
    reduction = reduction_[0]
    log.info(f'Reduction..........: {reduction}')

    modality_ = df['modality'].unique()
    assert modality_.shape[0] == 1, 'Multiple modalities are mixed. There should be only one'
    modality = modality_[0]
    log.info(f'Modality...........: {modality}')

    # Set save directories
    save_dir = os.path.join(cfg.save_dir, f'{reduction}_{modality}')
    img_dir = os.path.join(save_dir, 'img')
    os.makedirs(img_dir, exist_ok=True)

    # Copy images
    df['img_path_temp'] = df.loc[:, 'img_path']
    for idx, row in tqdm(df.iterrows(), desc='Copy images', unit=' images'):
        img_name = f'{reduction}_{modality}_{row["series"]}_{row["component"]:03}.png'
        df.at[idx, 'img_path'] = os.path.join(img_dir, img_name)
        df.at[idx, 'img_name'] = img_name
        shutil.copy(
            src=df.loc[idx, 'img_path_temp'],
            dst=df.loc[idx, 'img_path'],
        )
    df.drop('img_path_temp', axis=1, inplace=True)

    df.to_excel(
        os.path.join(save_dir, 'metadata.xlsx'),
        sheet_name='Metadata',
        index=True,
        index_label='ID',
    )

    log.info('')
    log.info(f'Complete')


if __name__ == '__main__':
    main()
