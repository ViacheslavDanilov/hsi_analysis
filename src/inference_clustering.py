import logging
import os
from functools import partial
from typing import Dict, List, Tuple

import cv2
import hydra
import imutils
import numpy as np
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data.utils import (
    extract_body_part,
    extract_series_name,
    extract_study_name,
    extract_temperature,
    extract_time_stamp,
    get_dir_list,
    get_file_list,
    read_hsi,
)
from src.models.models import AblationSegmenter

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def rescale_box(  # TODO: remove if it is not needed
    box: List[int],
    input_size: List[int],
    output_size: List[int],
) -> List[int]:

    y_scale = output_size[0] / input_size[0]
    x_scale = output_size[1] / input_size[1]

    box_rescaled = [
        int(box[0] * x_scale),
        int(box[1] * y_scale),
        int(box[2] * x_scale),
        int(box[3] * y_scale),
    ]

    return box_rescaled


def segment_hsi(
    hsi_path: str,
    save_dir: str,
    modality: str = 'abs',
    model_name: str = 'mean_shift',
    norm_type: str = 'standard',
    box_offset: Tuple[int, int] = (0, 0),
    src_size: Tuple[int, int] = (744, 1000),
) -> List:

    hsi_path = 'data/raw/30_08_2019_test_01_liver/10_00_39_T6=110/SpecCube.dat'  # TODO: remove after implementation

    # Read HSI file
    hsi = read_hsi(
        path=hsi_path,
        modality=modality,
    )

    # Extract additional information
    study_name = extract_study_name(path=hsi_path)
    series_name = extract_series_name(path=hsi_path)
    body_part = extract_body_part(path=hsi_path)  # TODO: implement usage
    temperature_idx, temperature = extract_temperature(path=hsi_path)  # TODO: implement usage
    date, time = extract_time_stamp(path=hsi_path)  # TODO: implement usage

    # Initialize clustering model
    model = AblationSegmenter(model_name)

    # Segment the ablation area
    box = [659, 553, 708, 601]  # TODO: temporary solution. Change in the future
    save_dir_img = os.path.join(save_dir, model_name, study_name, series_name, 'img')
    save_dir_box = os.path.join(save_dir, model_name, study_name, series_name, 'box')
    os.makedirs(save_dir_img, exist_ok=True)
    os.makedirs(save_dir_box, exist_ok=True)
    for idx in range(hsi.shape[2]):

        img = hsi[:, :, idx]

        # Process source image
        img_size = img.shape[:-1] if len(img.shape) == 3 else img.shape
        if img_size != src_size:
            img = imutils.resize(img, height=src_size[0], inter=cv2.INTER_LINEAR)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img = cv2.equalizeHist(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Clustering
        box_img, box_mask = model(
            img=img,
            box=box,
            box_offset=box_offset,
            norm_type=norm_type,
        )

        # Get number of clusters
        unique_clusters = list(np.unique(box_mask))
        num_clusters = len(unique_clusters)
        wavelength = 500 + 5 * idx
        wavelength_id = idx + 1
        log.info(f'ID = {wavelength_id:03}, Wavelength = {wavelength}, Clusters = {num_clusters}')

        # Stack and save segmentation masks
        box_mask_rgb = model.label_to_rgb(box_mask)
        box_stack = np.hstack([box_img, box_mask_rgb])
        img_name = f'{wavelength_id:03}.png'
        save_path_img = os.path.join(save_dir_img, img_name)
        save_path_box = os.path.join(save_dir_box, img_name)
        cv2.imwrite(save_path_img, img)
        cv2.imwrite(save_path_box, box_stack)

    return []


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'config'),
    config_name='inference_clustering',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Filter the list of studied directories
    study_dirs = get_dir_list(
        data_dir=cfg.src_dir,
        include_dirs=cfg.include_dirs,
        exclude_dirs=cfg.exclude_dirs,
    )

    # Get list of HSI files
    hsi_paths = get_file_list(
        src_dirs=study_dirs,
        include_template='',
        ext_list='.dat',
    )
    log.info(f'HSI found..........: {len(hsi_paths)}')

    # Multiprocessing of HSI files
    processing_func = partial(
        segment_hsi,
        modality=cfg.modality,
        model_name=cfg.model_name,
        norm_type=cfg.norm_type,
        box_offset=tuple(cfg.box_offset),
        src_size=tuple(cfg.src_size),
        save_dir=cfg.save_dir,
    )
    result = Parallel(n_jobs=-1, prefer='threads')(
        delayed(processing_func)(group) for group in tqdm(hsi_paths, desc='Clustering', unit=' HSI')
    )
    metadata: List[Dict] = sum(result, [])

    log.info('')
    log.info(f'Complete')


if __name__ == '__main__':
    main()
