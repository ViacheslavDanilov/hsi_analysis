import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.utils import get_file_list
from src.models.models import AblationDetector

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


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
    log.info('')
    log.info(f'Images to predict..: {len(img_paths)}')

    # Initialize model and load its weights
    detector = AblationDetector(
        model_dir=cfg.model_dir,
        conf_threshold=cfg.conf_threshold,
        device=cfg.device,
    )

    # Make predictions and get raw detections
    detections = detector.predict(
        img_paths=img_paths,
    )

    # Process predictions
    df = detector.process_detections(
        img_paths=img_paths,
        detections=detections,
    )

    # Save prediction metadata
    os.makedirs(cfg.save_dir, exist_ok=True)
    save_path = os.path.join(cfg.save_dir, 'predictions.xlsx')
    df.index += 1
    df.to_excel(
        save_path,
        sheet_name='Predictions',
        index=True,
        index_label='ID',
    )


if __name__ == '__main__':
    main()
