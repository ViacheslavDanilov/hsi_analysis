import json
import argparse

import numpy as np
from tqdm import tqdm
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

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


def crop(
        input_img: np.ndarray,
        img_type: str = 'absorbance',
) -> np.ndarray:

    assert input_img.shape[1] % 3 == 0, 'Input image width should be divisible by 3 i.e. contain 3 sub-images'

    img_width = int(input_img.shape[1] / 3)
    if img_type == 'absorbance':
        idx = 0
    elif img_type == 'hsv':
        idx = 1
    elif img_type == 'reflectance':
        idx = 2
    else:
        raise ValueError(f'Invalid img_type: {img_type}')

    output_img = input_img[:, idx * img_width:(idx + 1) * img_width]

    return output_img



def main(
        df: pd.DataFrame,
        class_names: Tuple[str],
        exclude_empty_masks: bool,
        use_smoothing: bool,
        img_type: str,
        save_dir: str,
) -> None:

    logger.info(f'Classes..............: {class_names}')
    logger.info(f'Number of classes....: {len(class_names)}')
    logger.info(f'Exclude empty masks..: {exclude_empty_masks}')
    logger.info(f'Output directory.....: {save_dir}')

    tests = list(set(df['test']))
    logger.info(f'Number of tests......: {len(tests)}')

    # FIXME: Split dataset into train, val and test subsets
    # Iterate over tests
    for test in tests:
        df_test = df[df['test'] == test]

        # # Iterate over videos
        for idx, row in tqdm(df_test.iterrows(), desc='Data conversion', unit=' video'):
            video_path = row['video_path']
            stem = row['stem']

            # Create image and annotation directories
            img_dir = os.path.join(save_dir, test, stem, 'img')
            os.makedirs(img_dir, exist_ok=True)

            # Read video and save images
            # TODO: uncomment later
            # video = cv2.VideoCapture(video_path)
            # count = 0
            # while True:
            #     success, _img = video.read()
            #     if not success:
            #         break
            #     img = crop(
            #         input_img=_img,
            #         img_type=img_type
            #     )
            #     img_path = os.path.join(img_dir, f'image_{count:02d}.png')
            #     cv2.imwrite(img_path, img)
            #     count += 1

            # TODO: Read json and save masks
            ann_path = row['ann_path']
            ann_dir = os.path.join(save_dir, test, stem, 'ann')
            os.makedirs(ann_dir, exist_ok=True)

            f = open(ann_path)
            ann_data = json.load(f)
            img_height, img_width = (
                ann_data['size']['height'],
                ann_data['size']['width'],
            )

    # TODO: Iterate over image-annotation pairs
    data = np.array([], dtype=np.uint8)
    for idx, row in tqdm(df_final.iterrows(), desc='Image processing', unit=' images'):
        filename = row['filename']
        img_path = row['img_path']
        ann_path = row['ann_path']
        f = open(ann_path)
        ann_data = json.load(f)
        img_size = (
            ann_data['size']['height'],
            ann_data['size']['width'],
        )

        # Iterate over objects
        mask = np.zeros(img_size, dtype=np.uint8)
        palette = get_palette(class_names)

        for obj in ann_data['objects']:
            class_name = obj['classTitle']
            if class_name in class_names:
                class_id = class_names.index(class_name)
                obj_mask64 = obj['bitmap']['data']
                obj_mask = base64_to_mask(obj_mask64)
                if use_smoothing:
                    obj_mask = smooth_mask(obj_mask)
                obj_mask = obj_mask.astype(float)
                obj_mask *= class_id/np.max(obj_mask)
                obj_mask = obj_mask.astype(np.uint8)

                mask = insert_mask(
                    mask=mask,
                    obj_mask=obj_mask,
                    origin=obj['bitmap']['origin'],
                )
        logger.debug('Empty mask: {:s}'.format(Path(filename).name))

        if (
                np.sum(mask) == 0
                and exclude_empty_masks
        ):
            pass
        else:
            subset = row['subset']

            # Save image
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img_save_path = os.path.join(save_dir, 'img_dir', subset, '{:s}.png'.format(filename))
            cv2.imwrite(img_save_path, img)

            # Accumulate masks for class weight calculation
            mask_vector = np.concatenate(mask)
            data = np.append(data, mask_vector, axis=0)

            # Save colored mask
            mask = Image.fromarray(mask).convert('P')
            mask.putpalette(np.array(palette, dtype=np.uint8))
            mask_save_path = os.path.join(save_dir, 'ann_dir', subset, '{:s}.png'.format(filename))
            mask.save(mask_save_path)

    # Compute class weights
    class_weights = class_weight.compute_class_weight(
        y=data,
        classes=np.unique(data),
        class_weight='balanced',
    )
    class_weights = list(class_weights)
    class_weights = [round(x, 3) for x in class_weights]
    logger.info('')
    logging.info('Class weights: {}'.format(class_weights))
    class_weights_path = os.path.join(save_dir, 'class_weights.txt')
    with open(class_weights_path, 'w') as f:
        f.write(str(class_names) + '\n')
        f.write(str(tuple(class_weights)))


if __name__ == '__main__':

    CLASSES = (
        'Background',
        'Core',
        'Ring',
    )

    parser = argparse.ArgumentParser(description='Dataset conversion')
    parser.add_argument('--project_dir', required=True, type=str)
    parser.add_argument('--annotator', default='ViacheslavDanilov', type=str, choices=['MartinaDL', 'ViacheslavDanilov'])
    parser.add_argument('--class_names', nargs='+', default=CLASSES, type=str)
    parser.add_argument('--exclude_empty_masks', action='store_true')
    parser.add_argument('--use_smoothing', action='store_true')
    parser.add_argument('--img_type',  default='absorbance', type=str, choices=['absorbance', 'hsv', 'reflectance'])
    parser.add_argument('--include_dirs', nargs='+', default=None, type=str)
    parser.add_argument('--exclude_dirs', nargs='+', default=None, type=str)
    parser.add_argument('--save_dir', required=True, type=str)
    args = parser.parse_args()

    df = read_sly_project(
        project_dir=args.project_dir,
        include_dirs=args.include_dirs,
        exclude_dirs=args.exclude_dirs,
    )

    main(
        df=df,
        class_names=tuple(args.class_names),
        exclude_empty_masks=args.exclude_empty_masks,
        use_smoothing=args.use_smoothing,
        img_type=args.img_type,
        save_dir=args.save_dir,
    )

    logger.info('')
    logger.info('Dataset conversion complete')
