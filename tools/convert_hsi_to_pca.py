import os
import logging
import argparse
from pathlib import Path
from functools import partial
from joblib import Parallel, delayed
from typing import List, Union, Tuple, Optional

import cv2
import imutils
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer

from tools.utils import (
    read_hsi,
    get_file_list,
    get_dir_list,
    get_study_name,
    get_series_name,
    get_color_map,
    extract_body_part,
    extract_temperature,
    extract_time_stamp,
)

os.makedirs('../logs', exist_ok=True)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d.%m.%Y %H:%M:%S',
    filename='logs/{:s}.log'.format(Path(__file__).stem),
    filemode='w',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def process_hsi(
        hsi_path: str,
        save_dir: str,
        num_components: int = 10,
        scaling_method: str = 'Standard',
        modality: str = 'absorbance',
        color_map: Optional[str] = None,
        apply_equalization: bool = False,
        output_size: Tuple[int, int] = (744, 1000),
) -> List:

    # Extract meta information
    study_name = get_study_name(path=hsi_path)
    series_name = get_series_name(path=hsi_path)
    body_part = extract_body_part(path=hsi_path)
    temperature_idx, temperature = extract_temperature(path=hsi_path)
    date, time = extract_time_stamp(path=hsi_path)

    # Read HSI file
    hsi_ = read_hsi(
        path=hsi_path,
        modality=modality,
    )
    hsi_height, hsi_width, hsi_bands = hsi_.shape
    hsi = hsi_.reshape(hsi_height * hsi_width, hsi_bands)
    X_ = pd.DataFrame(hsi)
    if scaling_method == 'Raw':
        X = np.array(X_, copy=True)
    elif scaling_method == 'MinMax':
        X = MinMaxScaler().fit_transform(X_)
    elif scaling_method == 'Standard':
        X = StandardScaler().fit_transform(X_)
    elif scaling_method == 'Robust':
        X = RobustScaler().fit_transform(X_)
    elif scaling_method == 'Power':
        X = PowerTransformer().fit_transform(X_)
    else:
        raise ValueError('Unsupported scaling')

    columns = [f'PC{idx + 1}' for idx in range(num_components)]
    pca = PCA(
        n_components=num_components,
        svd_solver='full',        # full arpack randomized
        random_state=11,
    )
    components = pca.fit_transform(X)
    var_ratio = list(pca.explained_variance_ratio_)

    # Process HSI in an image-by-image fashion
    save_dir = os.path.join(save_dir, study_name, series_name)
    os.makedirs(save_dir, exist_ok=True)
    metadata = []
    for idx in range(num_components):

        img = components[:, idx]
        img = img.reshape(hsi_height, hsi_width)
        img_name = f'{idx+1:03d}.png'
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
                'PC': idx + 1,
                'Variance': var_ratio[idx],
            }
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


def main(
        input_dir: str,
        save_dir: str,
        num_components: int = 10,
        scaling_method: str = 'Standard',
        modality: str = 'absorbance',
        color_map: Optional[str] = None,
        apply_equalization: bool = False,
        output_size: Tuple[int, int] = (744, 1000),
        include_dirs: Optional[Union[List[str], str]] = None,
        exclude_dirs: Optional[Union[List[str], str]] = None,
) -> None:

    # Log main parameters
    logger.info(f'Input dir..........: {input_dir}')
    logger.info(f'Included dirs......: {include_dirs}')
    logger.info(f'Excluded dirs......: {exclude_dirs}')
    logger.info(f'Components.........: {num_components}')
    logger.info(f'Scaling............: {scaling_method}')
    logger.info(f'Modality...........: {modality}')
    logger.info(f'Color map..........: {color_map}')
    logger.info(f'Apply equalization.: {apply_equalization}')
    logger.info(f'Output size........: {output_size}')
    logger.info(f'Output dir.........: {save_dir}')
    logger.info('')

    # Filter the list of studied directories
    study_dirs = get_dir_list(
        data_dir=input_dir,
        include_dirs=include_dirs,
        exclude_dirs=exclude_dirs,
    )

    # Get list of HSI files
    hsi_paths = get_file_list(
        src_dirs=study_dirs,
        include_template='',
        ext_list='.dat',
    )
    logger.info(f'HSI found..........: {len(hsi_paths)}')

    # Multiprocessing of HSI files
    os.makedirs(save_dir, exist_ok=True)
    processing_func = partial(
        process_hsi,
        num_components=num_components,
        scaling_method=scaling_method,
        modality=modality,
        color_map=color_map,
        apply_equalization=apply_equalization,
        output_size=output_size,
        save_dir=save_dir,
    )
    result = Parallel(n_jobs=-1)(
        delayed(processing_func)(group) for group in tqdm(hsi_paths, desc='Process hyperspectral images', unit='HSI')
    )
    metadata = sum(result, [])

    # Save metadata as an XLSX file
    df = pd.DataFrame(metadata)
    df.sort_values('Source dir', inplace=True)
    df.sort_index(inplace=True)
    save_path = os.path.join(save_dir, 'metadata.xlsx')
    df.index += 1
    df.to_excel(
        save_path,
        sheet_name='PCA',
        index=True,
        index_label='ID',
    )
    logger.info('')
    logger.info(f'Complete')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Apply PCA to HSI cubes')
    parser.add_argument('--input_dir', default='dataset/HSI', type=str)
    parser.add_argument('--include_dirs', nargs='+', default=None, type=str)
    parser.add_argument('--exclude_dirs', nargs='+', default=None, type=str)
    parser.add_argument('--num_components', default=10, type=int)
    parser.add_argument('--scaling_method', default='Standard', type=str, choices=['Raw', 'MinMax', 'Standard', 'Robust', 'Power'])
    parser.add_argument('--modality',  default='absorbance', type=str, choices=['absorbance', 'reflectance'])
    parser.add_argument('--color_map', default=None, type=str, choices=['jet', 'bone', 'ocean', 'cool', 'hsv'])
    parser.add_argument('--apply_equalization', action='store_true')
    parser.add_argument('--output_size', default=[744, 1000], nargs='+', type=int)
    parser.add_argument('--save_dir', default='dataset/PCA/HSI_abs', type=str)
    args = parser.parse_args()

    main(
        input_dir=args.input_dir,
        include_dirs=args.include_dirs,
        exclude_dirs=args.exclude_dirs,
        scaling_method=args.scaling_method,
        modality=args.modality,
        num_components=args.num_components,
        color_map=args.color_map,
        apply_equalization=args.apply_equalization,
        output_size=tuple(args.output_size),
        save_dir=args.save_dir,
    )
