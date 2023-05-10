import logging
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import hydra
import imutils
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageFilter
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


def filter_mask(
    mask: np.ndarray,
) -> np.ndarray:
    unique_clusters = list(np.unique(mask))
    unique_clusters.remove(0)

    mask_out = np.zeros_like(mask)
    for cluster_idx in unique_clusters:
        mask_idx = (mask == cluster_idx).astype(np.uint8)

        # Morphological smoothing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        _mask_morph = cv2.morphologyEx(mask_idx, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Filtering
        _mask_smooth = Image.fromarray(_mask_morph)
        _mask_smooth = _mask_smooth.filter(ImageFilter.ModeFilter(size=3))
        _mask_smooth = np.asarray(_mask_smooth)

        # Stacking layers
        mask_out_ = _mask_smooth * cluster_idx
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if mask_out_[y][x] == cluster_idx:
                    mask_out[y][x] = mask_out_[y][x]

    return mask_out


def segment_hsi(
    hsi_path: str,
    meta: pd.DataFrame,
    save_dir: str,
    modality: str = 'abs',
    model_name: str = 'mean_shift',
    norm_type: str = 'standard',
    box_offset: Tuple[int, int] = (0, 0),
    apply_filtering: bool = True,
    src_size: Tuple[int, int] = (744, 1000),
) -> List:
    # Read HSI file
    hsi = read_hsi(
        path=hsi_path,
        modality=modality,
    )

    # Extract additional information
    study_name = extract_study_name(path=hsi_path)
    series_name = extract_series_name(path=hsi_path)
    body_part = extract_body_part(path=hsi_path)
    temperature_idx, temperature = extract_temperature(path=hsi_path)
    date, time = extract_time_stamp(path=hsi_path)

    # Initialize clustering model
    model = AblationSegmenter(model_name)

    # Segment the ablation area
    metadata = []
    box_ = meta[(meta['study'] == study_name) & (meta['series'] == series_name)]
    box_ = box_[box_[['x1', 'y1', 'x2', 'y2']].notnull().all(1)]

    if len(box_) == 0:
        print('Status...............: Skipped')
        log.info('Status...............: Skipped')
    else:
        box = [
            int(np.mean(box_.x1)),
            int(np.mean(box_.y1)),
            int(np.mean(box_.x2)),
            int(np.mean(box_.y2)),
        ]
        save_dir_img = os.path.join(
            save_dir,
            model_name,
            study_name,
            series_name,
            f'img_{modality}',
        )
        save_dir_box_src = os.path.join(
            save_dir,
            model_name,
            study_name,
            series_name,
            f'box_{modality}_src',
        )
        save_dir_box_seg = os.path.join(
            save_dir,
            model_name,
            study_name,
            series_name,
            f'box_{modality}_seg',
        )
        os.makedirs(save_dir_img, exist_ok=True)
        os.makedirs(save_dir_box_src, exist_ok=True)
        os.makedirs(save_dir_box_seg, exist_ok=True)

        for idx in tqdm(range(hsi.shape[2]), leave=True, desc='Wavelength processing'):
            img = hsi[:, :, idx]

            # Process source image
            img_size = img.shape[:-1] if len(img.shape) == 3 else img.shape
            if img_size != src_size:
                img = imutils.resize(img, height=src_size[0], inter=cv2.INTER_LINEAR)
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            img = cv2.equalizeHist(img)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # Clustering
            box_src, box_mask = model(
                img=img,
                box=box,
                box_offset=box_offset,
                norm_type=norm_type,
            )

            if apply_filtering:
                box_mask = filter_mask(box_mask)

            # Get number of clusters
            unique_clusters = list(np.unique(box_mask))
            num_clusters = len(unique_clusters)
            wavelength = 500 + 5 * idx
            wavelength_id = idx + 1
            log.info(
                f'Study = {study_name}, '
                f'Series = {series_name}, '
                f'ID = {wavelength_id:03}, '
                f'Wavelength = {wavelength}, '
                f'Clusters = {num_clusters}',
            )

            # Stack and save segmentation masks
            box_seg = model.label_to_rgb(box_mask)
            img_name = f'{wavelength_id:03}.png'
            save_path_img = os.path.join(save_dir_img, img_name)
            save_path_box_src = os.path.join(save_dir_box_src, img_name)
            save_path_box_seg = os.path.join(save_dir_box_seg, img_name)
            cv2.imwrite(save_path_img, img)
            cv2.imwrite(save_path_box_src, box_src)
            cv2.imwrite(save_path_box_seg, box_seg)

            metadata.append(
                {
                    'source_dir': str(Path(hsi_path).parent),
                    'study_name': study_name,
                    'series_name': series_name,
                    'hsi_name': str(Path(hsi_path).name),
                    'hsi_path': hsi_path,
                    'hsi_height': hsi.shape[0],
                    'hsi_width': hsi.shape[1],
                    'hsi_depth': hsi.shape[2],
                    'img_name': img_name,
                    'img_path': save_path_img,
                    'img_height': img.shape[0],
                    'img_width': img.shape[1],
                    'box_name': img_name,
                    'box_path_src': save_path_box_src,
                    'box_path_seg': save_path_box_seg,
                    'box_height': box_src.shape[0],
                    'box_width': box_src.shape[1],
                    'date': date,
                    'time': time,
                    'body_part': body_part,
                    'temperature_id': temperature_idx,
                    'temperature': temperature,
                    'wavelength_id': idx + 1,
                    'wavelength': 500 + 5 * idx,
                    'clusters': unique_clusters,
                    'num_clusters': num_clusters,
                },
            )
        print('Status...............: Processed')
        log.info('Status...............: Processed')

    return metadata


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
    log.info(f'HSI found............: {len(hsi_paths)}')

    # Read metadata
    df_meta = pd.read_excel(cfg.metadata_path)

    # Processing of HSI files
    result = []
    for hsi_path in tqdm(hsi_paths, desc='HSI processing'):
        hsi_path = os.path.normpath(hsi_path)
        print(f'\nHSI to process.......: {hsi_path}')
        log.info(f'HSI to process.......: {hsi_path}')
        result_ = segment_hsi(
            hsi_path=hsi_path,
            meta=df_meta,
            modality=cfg.modality,
            model_name=cfg.model_name,
            norm_type=cfg.norm_type,
            box_offset=tuple(cfg.box_offset),
            src_size=tuple(cfg.src_size),
            apply_filtering=cfg.apply_filtering,
            save_dir=cfg.save_dir,
        )
        result.append(result_)
    result = sum(result, [])

    df = pd.DataFrame(result)
    save_path = os.path.join(cfg.save_dir, cfg.model_name, f'metadata_{cfg.modality}.xlsx')
    df.index += 1
    df.to_excel(
        save_path,
        sheet_name='Metadata',
        index=True,
        index_label='ID',
    )
    log.info('')
    log.info(f'Complete')


if __name__ == '__main__':
    main()
