import json
import argparse

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
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)


def log_dataset(
        dataset_name: str,
        train_images: int,
        val_images: int,
        test_images: int,
) -> None:
    logger.info('')
    logger.info('Dataset      : {:s}'.format(dataset_name))
    logger.info('Total images : {:d}'.format(train_images + val_images + test_images))
    logger.info('Train images : {:d}'.format(train_images))
    logger.info('Val images   : {:d}'.format(val_images))
    logger.info('Test images  : {:d}'.format(test_images))


def main(
        df_project: pd.DataFrame,
        class_names: Tuple[str],
        exclude_empty_masks: bool,
        use_smoothing: bool,
        train_size: float,
        test_size: float,
        seed: int,
        save_dir: str,
) -> None:

    val_size = 1 - train_size - test_size
    assert train_size + val_size + test_size == 1, 'The sum of subset ratios must be equal to 1'

    logger.info('Classes              : {}'.format(class_names))
    logger.info('Number of classes    : {}'.format(len(class_names)))
    logger.info('Exclude empty masks  : {}'.format(exclude_empty_masks))
    logger.info('Output directory     : {}'.format(save_dir))
    logger.info('Train/Val/Test split : {:.2f} / {:.2f} / {:.2f}'.format(train_size, val_size, test_size))
    datasets = list(set(df_project.dataset))

    # Split dataset into train, val and test subsets
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    df_test = pd.DataFrame()
    for dataset in datasets:
        df_dataset = df_project[df_project.dataset == dataset]
        df_dataset.reset_index(drop=True, inplace=True)
        _df_train, df_val_test = train_test_split(df_dataset, train_size=train_size, random_state=seed)
        _df_val, _df_test = train_test_split(df_val_test, test_size=test_size/(val_size + test_size), random_state=seed)
        log_dataset(dataset, len(_df_train), len(_df_val), len(_df_test))
        df_train = pd.concat([df_train, _df_train])
        df_val = pd.concat([df_val, _df_val])
        df_test = pd.concat([df_test, _df_test])

    df_train.sort_values(by=['img_path'], inplace=True)
    df_train.reset_index(drop=True, inplace=True)
    df_train['subset'] = 'train'

    df_val.sort_values(by=['img_path'], inplace=True)
    df_val.reset_index(drop=True, inplace=True)
    df_val['subset'] = 'val'

    df_test.sort_values(by=['img_path'], inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    df_test['subset'] = 'test'

    log_dataset('Final', len(df_train), len(df_val), len(df_test))

    os.makedirs(save_dir) if not os.path.isdir(save_dir) else False
    df_final = pd.concat([df_train, df_val, df_test], axis=0)
    df_final.reset_index(drop=True, inplace=True)
    save_path = os.path.join(save_dir, 'dataset.xlsx')
    df_final.to_excel(save_path, sheet_name='Dataset', index=False, startrow=0, startcol=0)
    logger.info('Dataset metadata save to {:s}'.format(save_path))

    # Create subset dirs
    subset_dirs = {
        'img_dir': {},
        'ann_dir': {},
    }
    for ds_dir in subset_dirs:
        for subset_dir in ['train', 'val', 'test']:
            _dir = os.path.join(save_dir, ds_dir, subset_dir)
            subset_dirs[ds_dir][subset_dir] = _dir
            os.makedirs(_dir) if not os.path.isdir(_dir) else False

    # Iterate over image-annotation pairs
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
    parser.add_argument('--class_names', nargs='+', default=CLASSES, type=str)
    parser.add_argument('--exclude_empty_masks', action='store_true')
    parser.add_argument('--use_smoothing', action='store_true')
    parser.add_argument('--include_dirs', nargs='+', default=None, type=str)
    parser.add_argument('--exclude_dirs', nargs='+', default=None, type=str)
    parser.add_argument('--train_size', default=0.80, type=float)
    parser.add_argument('--test_size', default=0.10, type=float)
    parser.add_argument('--seed', default=11, type=int)
    parser.add_argument('--save_dir', required=True, type=str)
    args = parser.parse_args()

    # FIXME: read project with video rather than images
    df = read_sly_project(
        project_dir=args.project_dir,
        include_dirs=args.include_dirs,
        exclude_dirs=args.exclude_dirs,
    )

    main(
        df_project=df,
        class_names=tuple(args.class_names),
        exclude_empty_masks=args.exclude_empty_masks,
        use_smoothing=args.use_smoothing,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
        save_dir=args.save_dir,
    )

    logger.info('')
    logger.info('Dataset conversion complete')
