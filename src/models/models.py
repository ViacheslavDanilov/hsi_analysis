import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from cpuinfo import get_cpu_info
from mmdet.apis import inference_detector, init_detector

from src.data.utils import get_file_list


class AblationDetector:
    """A class used during the inference of the detection pipeline."""

    def __init__(
        self,
        model_dir: str,
        conf_threshold: float = 0.01,
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
        self.classes = self.model.CLASSES
        self.model.test_cfg.rcnn.score_thr = conf_threshold

        # Log the device that is used for the prediction
        logging.info(f'Device......:')
        if device_ == 'cuda':
            logging.info(f'GPU.........: {torch.cuda.get_device_name(0)}')
            logging.info(f'Allocated...: {round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)} Gb')
            logging.info(f'Cached......: {round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)} Gb')
        else:
            info = get_cpu_info()
            logging.info(f'CPU.........: {info["brand_raw"]}')

    def predict(
        self,
        img_paths: List[str],
    ) -> List[List[np.ndarray]]:
        detections = inference_detector(
            model=self.model,
            imgs=img_paths,
        )

        return detections

    def process_detections(
        self,
        img_paths: List[str],
        detections: List[List[np.ndarray]],
    ) -> pd.DataFrame:

        columns = [
            'img_path',
            'img_name',
            'img_height',
            'img_width',
            'x1',
            'y1',
            'x2',
            'y2',
            'class_id',
            'class',
            'confidence',
        ]

        # Iterate over images
        df = pd.DataFrame(columns=columns)
        for image_idx, (img_path, detections_image) in enumerate(zip(img_paths, detections)):

            # Iterate over class detections
            for class_idx, detections_class in enumerate(detections_image):
                if detections_class.size == 0:
                    num_detections = 1
                else:
                    num_detections = detections_class.shape[0]

                # Iterate over boxes on a single image
                df_ = pd.DataFrame(index=range(num_detections), columns=columns)
                df_['img_path'] = img_path
                df_['img_name'] = Path(img_path).name
                for idx, box in enumerate(detections_class):
                    # box -> array(x_min, y_min, x_max, y_max, confidence)
                    df_.at[idx, 'x1'] = int(box[0])
                    df_.at[idx, 'y1'] = int(box[1])
                    df_.at[idx, 'x2'] = int(box[2])
                    df_.at[idx, 'y2'] = int(box[3])
                    df_.at[idx, 'class_id'] = class_idx
                    df_.at[idx, 'class'] = self.classes[class_idx]
                    df_.at[idx, 'confidence'] = box[4]
                df = pd.concat([df, df_])

        df.sort_values('img_path', inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df
