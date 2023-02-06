import logging
import multiprocessing
import os
from functools import partial
from typing import Tuple

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from src.data.utils import get_dir_list

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def convert_single_study(
    study_dir: str,
    output_type: str,
    output_size: Tuple[int, int],
    to_gray: bool,
    fps: int,
    save_dir: str,
) -> None:
    print('')


@hydra.main(config_path=os.path.join(os.getcwd(), 'config'), config_name='data', version_base=None)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    study_list = get_dir_list(
        data_dir=cfg.convert.study_dir,
        include_dirs=cfg.convert.include_dirs,
        exclude_dirs=cfg.convert.exclude_dirs,
    )

    num_cores = multiprocessing.cpu_count()
    conversion_func = partial(
        convert_single_study,
        output_type=cfg.convert.output_type,
        output_size=cfg.convert.output_size,
        to_gray=cfg.convert.to_gray,
        fps=cfg.convert.fps,
        save_dir=cfg.convert.save_dir,
    )
    process_map(
        conversion_func,
        tqdm(study_list, desc='Convert studies', unit=' study'),
        max_workers=num_cores,
    )
    log.info('Complete')


if __name__ == '__main__':
    main()
