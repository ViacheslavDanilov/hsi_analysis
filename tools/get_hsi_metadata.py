import argparse
import multiprocessing

import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from tools.utils import *

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d.%m.%Y %I:%M:%S',
    filename='logs/{:s}.log'.format(Path(__file__).stem),
    filemode='w',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def process_hsi(
        file_path: str,
) -> dict:

    hsi = read_hsi(file_path)
    body_part = extract_body_part(file_path)
    date, time = extract_time_stamp(filename=Path(file_path).name)
    test_dir = Path(file_path).parents[2].name

    metadata = {
        'Source dir': str(Path(file_path).parent),
        'Test name': str(test_dir),
        'HSI name': Path(file_path).name,
        'Date': date,
        'Time': time,
        'Body part': body_part,
        'Height': hsi.shape[0],
        'Width': hsi.shape[1],
        'Depth': hsi.shape[2],
        'Size': os.path.getsize(file_path) / (10**6),
    }
    logger.info(f'Metadata extracted.: {Path(file_path).name}')

    return metadata


def main(
        input_dir: str,
        save_dir: str,
        include_dirs: Optional[Union[List[str], str]] = None,
        exclude_dirs: Optional[Union[List[str], str]] = None,
):
    # Log main parameters
    logger.info(f'Input dir..........: {input_dir}')
    logger.info(f'Included dirs......: {include_dirs}')
    logger.info(f'Excluded dirs......: {exclude_dirs}')
    logger.info(f'Output dir.........: {save_dir}')
    logger.info('')

    # Filter the list of studied directories
    study_dirs = get_dir_list(
        data_dir=input_dir,
        include_dirs=include_dirs,
        exclude_dirs=exclude_dirs,
    )

    # Get list of HSI files
    hsi_paths = get_file_list(
        src_dirs=study_dirs,
        include_template='',
        ext_list='.dat',
    )
    logger.info(f'HSI found..........: {len(hsi_paths)}')

    # Multiprocessing of all HSIs
    num_cores = multiprocessing.cpu_count()
    metadata = process_map(
        process_hsi,
        tqdm(hsi_paths, desc='Process hyperspectral images', unit=' HSI'),
        max_workers=num_cores,
    )

    # Save metadata as an XLSX file
    df = pd.DataFrame(metadata)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'HSI metadata.xlsx')
    df.index += 1
    df.to_excel(
        save_path,
        sheet_name='Metadata',
        index=True,
        index_label='ID',
    )

    logger.info('')
    logger.info(f'Complete')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Data processing')
    parser.add_argument('--input_dir', default='dataset/source', type=str)
    parser.add_argument('--include_dirs', nargs='+', default=None, type=str)
    parser.add_argument('--exclude_dirs', nargs='+', default=None, type=str)
    parser.add_argument('--save_dir', default='calculations', type=str)
    args = parser.parse_args()

    main(
        input_dir=args.input_dir,
        include_dirs=args.include_dirs,
        exclude_dirs=args.exclude_dirs,
        save_dir=args.save_dir,
    )
