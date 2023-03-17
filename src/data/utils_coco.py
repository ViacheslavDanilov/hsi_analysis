from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import supervisely_lib as sly

from src.data.utils_sly import CLASS_MAP


def get_img_info(
    img_path: str,
    img_id: int,
) -> Dict[str, Any]:
    img_data: Dict[str, Union[int, str]] = {}
    height, width = cv2.imread(img_path).shape[:2]
    img_data['id'] = img_id  # Unique image ID
    img_data['width'] = width
    img_data['height'] = height
    img_data['file_name'] = Path(img_path).name
    return img_data


# TODO: change usage of label path
def get_ann_info(
    label_path: str,
    img_id: int,
    ann_id: int,
    box_extension: dict,
) -> Tuple[List[Any], int]:

    ann = sly.io.json.load_json_file(label_path)

    ann_data = []
    for obj in ann['objects']:
        class_name = obj['classTitle']
        figure = sly.Rectangle.from_json(obj)

        box_extension_class = box_extension[class_name]
        x1, y1 = (
            figure.left - box_extension_class[0],
            figure.top - box_extension_class[1],
        )
        x2, y2 = (
            figure.right + box_extension_class[0],
            figure.bottom + box_extension_class[1],
        )
        width = abs(x2 - x1 + 1)
        height = abs(y2 - y1 + 1)

        label = {
            'id': ann_id,
            'image_id': img_id,
            'category_id': int(CLASS_MAP[class_name]),
            'bbox': [x1, y1, width, height],
            'area': int(width * height),
            'iscrowd': 0,
        }

        ann_data.append(label)
        ann_id += 1

    return ann_data, ann_id
