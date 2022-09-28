import os
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import supervisely_lib as sly


def read_sly_project(
    project_dir: str,
    include_dirs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
) -> pd.DataFrame:

    logging.info(f'Dataset dir..........: {project_dir}')
    assert os.path.exists(project_dir) and os.path.isdir(project_dir), 'Wrong project dir: {}'.format(project_dir)
    project = sly.VideoProject(
        directory=project_dir,
        mode=sly.OpenMode.READ,
    )

    series_list: List[str] = []
    study_list: List[str] = []
    video_paths: List[str] = []
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
            video_path, ann_path = dataset.get_item_paths(item_name)
            series = Path(video_path).stem
            series_list.append(series)
            video_paths.append(video_path)
            ann_paths.append(ann_path)
            study_list.append(study)

    df = pd.DataFrame.from_dict(
        {
            'study': study_list,
            'series': series_list,
            'video_path': video_paths,
            'ann_path': ann_paths,
        }
    )
    df.sort_values(['study', 'series'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
