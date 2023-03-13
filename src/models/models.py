import logging
from typing import List

import pandas as pd
import torch
from cpuinfo import get_cpu_info
from mmdet.apis import inference_detector, init_detector

from src.data.utils import get_file_list


class AblationDetector:
    """A class used during the inference of the ablation detection pipeline."""

    def __init__(
        self,
        model_dir: str,
        conf_threshold: float = 0.5,  # TODO: implement usage of this parameter
        iou_threshold: float = 0.3,  # TODO: implement usage of this parameter
        device: str = 'auto',
    ):

        # Get config path
        config_list = get_file_list(
            src_dirs=model_dir,
            ext_list='.py',
        )
        assert len(config_list) == 1, 'Keep only one config file in the model directory'
        config_path = config_list[0]

        # Get checkpoint path
        checkpoint_list = get_file_list(
            src_dirs=model_dir,
            ext_list='.pth',
        )
        assert len(checkpoint_list) == 1, 'Keep only one checkpoint file in the model directory'
        checkpoint_path = checkpoint_list[0]

        # Load the model
        if device == 'cpu':
            device_ = 'cpu'
        elif device == 'gpu':
            device_ = 'cuda'
        elif device == 'auto':
            device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            raise ValueError(f'Unknown device: {device}')

        self.model = init_detector(
            config=config_path,
            checkpoint=checkpoint_path,
            device=device_,
        )

        # Log the device that is used for the prediction
        logging.info(f'Device......:')
        if device_ == 'cuda':
            logging.info(f'GPU.........: {torch.cuda.get_device_name(0)}')
            logging.info(f'Allocated...: {round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)} Gb')
            logging.info(f'Cached......: {round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)} Gb')
        else:
            info = get_cpu_info()
            logging.info(f'CPU.........: {info["brand_raw"]}')

    # TODO: add method for forecasting
    def predict(
        self,
        img_paths: List[str],
    ) -> pd.DataFrame:
        # output -> array(x_min, y_min, x_max, y_max, confidence)
        detections = inference_detector(
            model=self.model,
            imgs=img_paths,
        )

        return detections  # FIXME: change type