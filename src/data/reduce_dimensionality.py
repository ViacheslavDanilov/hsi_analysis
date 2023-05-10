import logging
import os
import pickle
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import hydra
import imutils
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.data.utils import (
    extract_body_part,
    extract_series_name,
    extract_study_name,
    extract_temperature,
    extract_time_stamp,
    get_color_map,
    get_dir_list,
    get_file_list,
    read_hsi,
)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def reduce_dimensionality(
    data: np.ndarray,
    reduction_method: str = 'pca',
    num_components: int = 3,
) -> Tuple[np.ndarray, List[float]]:
    if reduction_method == 'tsne':
        tsne = TSNE(
            n_components=num_components,
            perplexity=30,
            n_iter=300,
            verbose=1,
        )
        data_reduced = tsne.fit_transform(data)
        var_ratio = [np.nan for x in range(num_components)]
    elif reduction_method == 'pca':
        pca = PCA(
            n_components=num_components,
            svd_solver='full',
            random_state=11,
        )
        data_reduced = pca.fit_transform(data)
        var_ratio = list(pca.explained_variance_ratio_)
    else:
        raise ValueError(f'Unknown reduction method: {reduction_method}')

    return data_reduced, var_ratio


def process_hsi(
    hsi_path: str,
    save_dir: str,
    num_components: int = 3,
    reduction_method: str = 'pca',
    modality: str = 'abs',
    color_map: Optional[str] = None,
    apply_equalization: bool = False,
    output_size: Tuple[int, int] = (744, 1000),
) -> List:
    # Extract meta information
    study_name = extract_study_name(path=hsi_path)
    series_name = extract_series_name(path=hsi_path)
    body_part = extract_body_part(path=hsi_path)
    temperature_idx, temperature = extract_temperature(path=hsi_path)
    date, time = extract_time_stamp(path=hsi_path)

    # Read HSI file
    hsi = read_hsi(
        path=hsi_path,
        modality=modality,
    )
    hsi_height, hsi_width, hsi_bands = hsi.shape
    hsi_reshaped = hsi.reshape(hsi_height * hsi_width, hsi_bands)

    # Normalize data
    hsi_norm = StandardScaler().fit_transform(hsi_reshaped)

    # Apply PCA or TSNE transformation
    pickle_dir = os.path.join(Path(save_dir).parent, f'{modality}_pkl', study_name)
    pickle_name = f'{series_name}.pkl'
    pickle_path = os.path.join(pickle_dir, pickle_name)
    os.makedirs(pickle_dir, exist_ok=True)
    if Path(pickle_path).exists():
        with open(pickle_path, 'rb') as f:
            hsi_reduced, var_ratio = pickle.load(f)
            log.info(f'Load reduced HSI: {pickle_path}')
            f.close()
    elif not Path(pickle_path).exists():
        hsi_reduced, var_ratio = reduce_dimensionality(
            data=hsi_norm,
            reduction_method=reduction_method,
            num_components=num_components,
        )
        with open(pickle_path, 'wb') as f:
            pickle.dump([hsi_reduced, var_ratio], f)
            log.info(f'Save reduced HSI: {pickle_path}')
            f.close()
    else:
        raise ValueError('Unexpected error appeared during reduction')

    # Process HSI in an image-by-image fashion
    save_dir = os.path.join(save_dir, study_name)
    os.makedirs(save_dir, exist_ok=True)
    metadata = []
    for idx in range(num_components):
        img = hsi_reduced[:, idx]
        img = img.reshape(hsi_height, hsi_width)
        img_name = f'{series_name}_{idx+1:03d}.png'
        img_path = os.path.join(save_dir, img_name)

        metadata.append(
            {
                'Source dir': str(Path(hsi_path).parent),
                'Study name': study_name,
                'Series name': series_name,
                'HSI name': str(Path(hsi_path).name),
                'Image name': img_name,
                'Image path': img_path,
                'Date': date,
                'Time': time,
                'Body part': body_part,
                'Min': np.min(img),
                'Mean': np.mean(img),
                'Max': np.max(img),
                'Height': img.shape[0],
                'Width': img.shape[1],
                'Temperature ID': temperature_idx,
                'Temperature': temperature,
                'Method': reduction_method,
                'PC': idx + 1,
                'Variance': var_ratio[idx],
            },
        )

        # Resize and normalize image
        img_size = img.shape[:-1] if len(img.shape) == 3 else img.shape
        if img_size != output_size:
            img = imutils.resize(img, height=output_size[0], inter=cv2.INTER_LINEAR)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Equalize histogram
        if apply_equalization:
            img = cv2.equalizeHist(img)

        # Colorize image
        if isinstance(color_map, str) and color_map is not None:
            cmap = get_color_map(color_map)
            img = cv2.applyColorMap(img, cmap)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        cv2.imwrite(img_path, img)

    return metadata


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'config'),
    config_name='reduce_dimensionality',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    if cfg.color_map is not None:
        save_dir = os.path.join(cfg.save_dir, cfg.reduction_method, cfg.color_map)
    else:
        save_dir = os.path.join(cfg.save_dir, cfg.reduction_method, cfg.modality)

    # Log main parameters
    log.info(f'Input dir..........: {cfg.src_dir}')
    log.info(f'Included dirs......: {cfg.include_dirs}')
    log.info(f'Excluded dirs......: {cfg.exclude_dirs}')
    log.info(f'Components.........: {cfg.num_components}')
    log.info(f'Reduction method...: {cfg.reduction_method}')
    log.info(f'Modality...........: {cfg.modality}')
    log.info(f'Color map..........: {cfg.color_map}')
    log.info(f'Apply equalization.: {cfg.apply_equalization}')
    log.info(f'Output size........: {cfg.output_size}')
    log.info(f'Output dir.........: {save_dir}')
    log.info('')

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
    os.makedirs(save_dir, exist_ok=True)
    processing_func = partial(
        process_hsi,
        num_components=cfg.num_components,
        reduction_method=cfg.reduction_method,
        modality=cfg.modality,
        color_map=cfg.color_map,
        apply_equalization=cfg.apply_equalization,
        output_size=cfg.output_size,
        save_dir=save_dir,
    )
    result = Parallel(n_jobs=-1, prefer='threads')(
        delayed(processing_func)(group)
        for group in tqdm(hsi_paths, desc='Reduce hyperspectral images', unit='HSI')
    )
    metadata: List[Dict] = sum(result, [])

    # Save metadata as an XLSX file
    df = pd.DataFrame(metadata)
    df.sort_values('Source dir', inplace=True)
    df.sort_index(inplace=True)
    save_path = os.path.join(save_dir, 'metadata.xlsx')
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
