import logging
import os
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
            'ann_path',
            'img_name',
            'ann_name',
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
            'ann_path': ann_path,
            'img_name': img_name,
            'ann_name': ann_name,
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
    df.sort_index(inplace=True)

    #     save_path = os.path.join(save_dir, 'metadata.xlsx')
    #     df.index += 1
    #     df.to_excel(
    #         save_path,
    #         sheet_name='Metadata',
    #         index=True,
    #         index_label='ID',
    #     )
    print('')

    #             img_save_path = os.path.join(save_img_dir, img_data['file_name'])
    #             shutil.copy(
    #                 src=sample['img_path'],
    #                 dst=img_save_path,
    #             )


if __name__ == '__main__':
    main()
