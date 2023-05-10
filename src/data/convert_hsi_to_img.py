import logging
import os
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import ffmpeg
import hydra
import imutils
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
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


def process_hsi(
    hsi_path: str,
    save_dir: str,
    modality: str = 'abs',
    color_map: str = None,
    apply_equalization: bool = False,
    output_type: str = 'image',
    output_size: Tuple[int, int] = (744, 1000),
    fps: int = 15,
) -> List:
    assert output_type == 'image' or output_type == 'video', f'Incorrect output_type: {output_type}'

    # Read HSI file and extract additional information
    hsi = read_hsi(
        path=hsi_path,
        modality=modality,
    )
    study_name = extract_study_name(path=hsi_path)
    series_name = extract_series_name(path=hsi_path)
    body_part = extract_body_part(path=hsi_path)
    temperature_idx, temperature = extract_temperature(path=hsi_path)
    date, time = extract_time_stamp(path=hsi_path)

    # Select output_dir based on output_type
    if output_type == 'image':
        output_dir = os.path.join(save_dir, study_name, series_name)
    else:
        output_dir = os.path.join(save_dir, study_name)
    os.makedirs(output_dir, exist_ok=True)

    # Create video writer
    if output_type == 'video':
        video_path_temp = os.path.join(output_dir, f'{series_name}_temp.mp4')
        _img = imutils.resize(hsi[:, :, 0], height=output_size[0], inter=cv2.INTER_LINEAR)
        _img_size = _img.shape[:-1] if len(_img.shape) == 3 else _img.shape
        video_height, video_width = _img_size
        video = cv2.VideoWriter(
            video_path_temp,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (video_width, video_height),
        )

    # Process HSI in an image-by-image fashion
    metadata = []
    for idx in range(hsi.shape[2]):
        img = hsi[:, :, idx]
        img_name = f'{idx+1:03d}.png'
        img_path = os.path.join(output_dir, img_name)

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
                'Wavelength ID': idx + 1,
                'Wavelength': 500 + 5 * idx,
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

        # Save image
        if output_type == 'image':
            cv2.imwrite(img_path, img)
        elif output_type == 'video':
            video.write(img)
        else:
            raise ValueError(f'Unknown output_type value: {output_type}')

    video.release() if output_type == 'video' else False

    # Replace OpenCV videos with FFmpeg ones
    if output_type == 'video':
        video_path = os.path.join(output_dir, f'{series_name}.mp4')
        stream = ffmpeg.input(video_path_temp)
        stream = ffmpeg.output(stream, video_path, vcodec='libx264', video_bitrate='10M')
        ffmpeg.run(stream, quiet=True, overwrite_output=True)
        os.remove(video_path_temp)

    log.info(f'HSI processed......: {hsi_path}')

    return metadata


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'config'),
    config_name='convert_hsi_to_img',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    if cfg.color_map is not None:
        save_dir = os.path.join(cfg.save_dir, cfg.color_map)
    else:
        save_dir = os.path.join(cfg.save_dir, cfg.modality)

    # Log main parameters
    log.info(f'Input dir..........: {cfg.src_dir}')
    log.info(f'Included dirs......: {cfg.include_dirs}')
    log.info(f'Excluded dirs......: {cfg.exclude_dirs}')
    log.info(f'Output dir.........: {save_dir}')
    log.info(f'Modality...........: {cfg.modality}')
    log.info(f'Color map..........: {cfg.color_map}')
    log.info(f'Apply equalization.: {cfg.apply_equalization}')
    log.info(f'Output type........: {cfg.output_type}')
    log.info(f'Output size........: {cfg.output_size}')
    log.info(f'FPS................: {cfg.fps if cfg.output_type == "video" else None}')
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
        modality=cfg.modality,
        color_map=cfg.color_map,
        apply_equalization=cfg.apply_equalization,
        output_type=cfg.output_type,
        output_size=tuple(cfg.output_size),
        fps=cfg.fps,
        save_dir=save_dir,
    )
    result = Parallel(n_jobs=-1, prefer='threads')(
        delayed(processing_func)(group)
        for group in tqdm(hsi_paths, desc='Process hyperspectral images', unit=' HSI')
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
