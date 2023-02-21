import logging
import multiprocessing
import os
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

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

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def process_hsi(
    hsi_path: str,
) -> dict:

    hsi = read_hsi(hsi_path)
    study_name = extract_study_name(path=hsi_path)
    series_name = extract_series_name(path=hsi_path)
    body_part = extract_body_part(path=hsi_path)
    date, time = extract_time_stamp(path=hsi_path)
    temperature_idx, temperature = extract_temperature(path=hsi_path)

    metadata = {
        'Source dir': str(Path(hsi_path).parent),
        'Study name': study_name,
        'Series name': series_name,
        'HSI name': str(Path(hsi_path).name),
        'Date': date,
        'Time': time,
        'Body part': body_part,
        'Height': hsi.shape[0],
        'Width': hsi.shape[1],
        'Depth': hsi.shape[2],
        'Size': os.path.getsize(hsi_path) / (10**6),
        'Temperature ID': temperature_idx,
        'Temperature': temperature,
    }
    log.info(f'Metadata extracted.: {hsi_path}')

    return metadata


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'config'),
    config_name='get_hsi_metadata',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Log main parameters
    log.info(f'Input dir..........: {cfg.src_dir}')
    log.info(f'Included dirs......: {cfg.include_dirs}')
    log.info(f'Excluded dirs......: {cfg.exclude_dirs}')
    log.info(f'Output dir.........: {cfg.save_dir}')
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

    # Multiprocessing of all HSIs
    num_cores = multiprocessing.cpu_count()
    metadata = process_map(
        process_hsi,
        tqdm(hsi_paths, desc='Process hyperspectral images', unit=' HSI'),
        max_workers=num_cores,
    )

    # Save metadata as an XLSX file
    df = pd.DataFrame(metadata)
    os.makedirs(cfg.save_dir, exist_ok=True)
    save_path = os.path.join(cfg.save_dir, 'metadata.xlsx')
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
