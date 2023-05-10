import logging
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
import supervisely_lib as sly

CLASS_MAP = {
    'Ablation': 1,
}

CLASS_MAP_REVERSED = dict((v, k) for k, v in CLASS_MAP.items())


def read_sly_project(
    project_dir: str,
    include_dirs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> pd.DataFrame:
    logging.info(f'Dataset dir..........: {project_dir}')
    assert os.path.exists(project_dir) and os.path.isdir(
        project_dir,
    ), 'Wrong project dir: {}'.format(project_dir)
    try:
        project = sly.VideoProject(
            directory=project_dir,
            mode=sly.OpenMode.READ,
        )
        project_type = 'video'
    except Exception as e:
        project = sly.Project(
            directory=project_dir,
            mode=sly.OpenMode.READ,
        )
        project_type = 'img'

    series_list: List[str] = []
    study_list: List[str] = []
    file_paths: List[str] = []
    ann_paths: List[str] = []

    for dataset in project:
        study = dataset.name

        if include_dirs and study not in include_dirs:
            logging.info(f'Excluded dir.........: {study}')
            continue

        if exclude_dirs and study in exclude_dirs:
            logging.info(f'Excluded dir.........: {study}')
            continue

        logging.info(f'Included dir.........: {study}')
        for item_name in dataset:
            file_path, ann_path = dataset.get_item_paths(item_name)
            series = Path(file_path).stem
            series_list.append(series)
            file_paths.append(file_path)
            ann_paths.append(ann_path)
            study_list.append(study)

    df = pd.DataFrame.from_dict(
        {
            'study': study_list,
            'series': series_list,
            f'{project_type}_path': file_paths,
            'ann_path': ann_paths,
        },
    )
    df.sort_values(['study', 'series'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
