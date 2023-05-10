import logging
import os

import fiftyone as fo
import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'config'),
    config_name='visualize_coco_dataset',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    subset_dir = os.path.join(cfg.dataset_dir, cfg.subset)
    try:
        dataset = fo.Dataset.from_dir(
            dataset_dir=subset_dir,
            dataset_type=fo.types.COCODetectionDataset,
            overwrite=True,
            persistent=False,
            max_samples=cfg.max_samples,
            shuffle=cfg.shuffle,
            seed=cfg.seed,
        )
    except ValueError:
        dataset = fo.load_dataset(cfg.dataset_name)
    session = fo.launch_app(dataset)
    session.wait()
    dataset.delete()


if __name__ == '__main__':
    main()
