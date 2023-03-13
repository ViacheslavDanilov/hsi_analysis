import logging
import os
from typing import List, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.data.utils import get_file_list
from src.models.models import AblationDetector

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def rescale_detections(
    detections: List[np.ndarray],
    src_size: Tuple[int, int],
    new_size: Tuple[int, int],
) -> List[np.ndarray]:

    for detections_per_class in detections:
        detections_per_class[:, 0] *= new_size[1] / src_size[1]  # x_min
        detections_per_class[:, 1] *= new_size[0] / src_size[0]  # y_min
        detections_per_class[:, 2] *= new_size[1] / src_size[1]  # x_max
        detections_per_class[:, 3] *= new_size[0] / src_size[0]  # y_max

    return detections


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'config'),
    config_name='inference',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    # Get list of images to predict
    img_paths = get_file_list(
        src_dirs=cfg.data_dir,
        ext_list=[
            '.png',
            '.jpg',
            '.jpeg',
            '.bmp',
        ],
    )

    # Initialize model and load its weights
    ablation_detector = AblationDetector(
        model_dir=cfg.model_dir,
        conf_threshold=cfg.conf_threshold,
        iou_threshold=cfg.iou_threshold,
        device=cfg.device,
    )

    detections = ablation_detector.predict(img_paths)


if __name__ == '__main__':
    main()
