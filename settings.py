# Supervisely to COCO
SLY_PROJECT_DIR = 'dataset/SLY'
COCO_SAVE_DIR = 'dataset/COCO'
INCLUDE_DIRS = []
EXCLUDE_DIRS = [
    '29_08_2019_test_08_pancreas',
    '30_08_2019_test_04_pancreas',
    '30_08_2019_test_06_pancreas',
    '30_08_2019_test_08_stomach',
]
TRAIN_SIZE = 0.80
SEED = 11
BOX_EXTENSION = {
    # class: (horizontal, vertical)
    'Ablation': (0, 0),
}
