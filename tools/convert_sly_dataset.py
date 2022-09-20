import json
import argparse
from functools import partial
from joblib import Parallel, delayed

import numpy as np
from tqdm import tqdm

from tools.utils import crop_image
from tools.supervisely_utils import *


os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d.%m.%Y %H:%M:%S',
    filename='logs/{:s}.log'.format(Path(__file__).stem),
    filemode='w',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def process_single_video(
        group: Tuple[int, pd.DataFrame],
        img_type: str,
        # dataset_dir: str,
        # save_pairs_only: bool,
        save_dir: str,
) -> pd.DataFrame:
    _, row = group
    video_path = row['video_path']
    stem = row['stem']
    test = row['test']

    # Create image and annotation directories
    img_dir = os.path.join(save_dir, test, stem, 'img')
    ann_dir = os.path.join(save_dir, test, stem, 'ann')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    # Read video, save images, and empty annotations
    # TODO: VERSION 1
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    while True:
        success, _img = video.read()
        if not success:
            break
        img = crop_image(
            input_img=_img,
            img_type=img_type
        )
        img_path = os.path.join(img_dir, f'{count+1:03d}.png')
        cv2.imwrite(img_path, img)

        txt_path = os.path.join(ann_dir, f'{count+1:03d}.txt')
        f = open(file=txt_path, mode='w')
        f.close()
        count += 1

    # Read json with metadata
    ann_path = row['ann_path']
    f = open(ann_path)
    ann_data = json.load(f)
    img_height = ann_data['size']['height']
    img_width = ann_data['size']['width'] // 2          # delimiter is a number of images in a stacked row

    # Extract object keys based on annotator name
    # TODO: remove after debugging bboxes
    key_obj_map = {}
    for obj in ann_data['objects']:
        key_obj_map[obj['classTitle']] = obj['key']

    # Save annotations
    for frame in ann_data['frames']:
        frame_idx = frame['index']
        # TODO: Iterate over figures
        for _figure in frame['figures']:

            figure = sly.Rectangle.from_json(_figure['geometry'])

            if figure.left > img_width:
                figure = figure.translate(drow=img_width, dcol=0)

    return 1


def main(
        df: pd.DataFrame,
        annotator: str,
        box_extension: int,
        img_type: str,
        save_dir: str,
) -> None:

    logger.info(f'Annotator............: {annotator}')
    logger.info(f'Box extension........: {box_extension}')
    logger.info(f'Output directory.....: {save_dir}')

    tests = list(set(df['test']))
    logger.info(f'Number of tests......: {len(tests)}')

    processing_func = partial(
        process_single_video,
        img_type=img_type,
        save_dir=save_dir,
    )
    # TODO: change to -1 after debugging
    result = Parallel(n_jobs=1)(
        delayed(processing_func)(row) for row in tqdm(df.iterrows(), desc='Dataset conversion', unit=' video')
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Dataset conversion')
    parser.add_argument('--project_dir', required=True, type=str)
    parser.add_argument('--include_dirs', nargs='+', default=None, type=str)
    parser.add_argument('--exclude_dirs', nargs='+', default=None, type=str)
    parser.add_argument('--annotator', default='ViacheslavDanilov', type=str, choices=['MartinaDL', 'ViacheslavDanilov'])
    parser.add_argument('--box_extension', default=10, type=int)
    parser.add_argument('--img_type',  default='absorbance', type=str, choices=['absorbance', 'hsv'])
    parser.add_argument('--save_dir', required=True, type=str)
    args = parser.parse_args()

    df = read_sly_project(
        project_dir=args.project_dir,
        include_dirs=args.include_dirs,
        exclude_dirs=args.exclude_dirs,
    )

    main(
        df=df,
        annotator=args.annotator,
        box_extension=args.box_extension,
        img_type=args.img_type,
        save_dir=args.save_dir,
    )

    logger.info('')
    logger.info('Dataset conversion complete')
