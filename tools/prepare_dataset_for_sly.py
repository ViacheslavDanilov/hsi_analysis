import os
import shutil
import logging
import argparse
from pathlib import Path

from tqdm import tqdm

from tools.utils import get_file_list, extract_study_name, extract_series_name

os.makedirs('../logs', exist_ok=True)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d.%m.%Y %H:%M:%S',
    filename='logs/{:s}.log'.format(Path(__file__).stem),
    filemode='w',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main(
    data_dir: str,
    save_dir: str,
) -> None:

    img_paths_src = get_file_list(
        src_dirs=data_dir,
        ext_list=[
            '.png',
            '.jpg',
            '.jpeg',
            '.bmp',
        ]
    )

    img_names = []
    for img_path_src in tqdm(img_paths_src, desc='Image preparation for Supervisely', unit=' images'):
        study_name = extract_study_name(img_path_src)
        series_name = extract_series_name(img_path_src)
        img_name = Path(img_path_src).name

        save_dir_dst = os.path.join(save_dir, study_name)
        os.makedirs(save_dir_dst, exist_ok=True)
        img_name_dst = f'{series_name}_{img_name}'
        img_path_dst = os.path.join(save_dir_dst, img_name_dst)
        shutil.copy(
            src=img_path_src,
            dst=img_path_dst,
        )
        img_names.append(img_name_dst)

    assert len(set(img_paths_src)) == len(set(img_names)), 'There are images with duplicate names'


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Reduce dimensionality of HSI cubes')
    parser.add_argument('--data_dir', default='dataset/TSNE/absorbance', type=str)
    parser.add_argument('--save_dir', default='dataset/TSNE/absorbance_sly', type=str)
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
    )
