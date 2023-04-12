# Segmentation and analysis of hyperspectral images
This repository is dedicated to the segmentation of [hyperspectral images](https://en.wikipedia.org/wiki/Hyperspectral_imaging) during experimental animal surgery.

## Requirements

- Linux or macOS (Windows has not been officially tested)
- Python 3.8.x

## Installation

Step 1: Download and install Miniconda
``` bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_22.11.1-1-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

Step 2: Install FFmpeg and verify that the installation is correct

- Linux
``` bash
sudo apt update
sudo apt upgrade
sudo apt install ffmpeg
ffmpeg -version
```

- macOS
``` bash
brew update
brew upgrade
brew install ffmpeg
ffmpeg
```

Step 3: Clone the repository, create a conda environment, and install the requirements for the repository
``` bash
git clone https://github.com/ViacheslavDanilov/oct_segmentation.git
cd oct_segmentation
chmod +x create_env.sh
source create_env.sh
```

Step 4: Initialize git hooks using the pre-commit framework
``` bash
pre-commit install
```

## Data


<p align="right">Table 1. Example of hyperspectral images taken at different wavelengths</p>

|                                                     Absorbance                                                      |                                                         HSV                                                         |                                                     Reflectance                                                     |
|:-------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------:|
| <video src="https://user-images.githubusercontent.com/32963687/231411783-c27c9b29-bfc0-4795-a804-a4b4ee17e15f.mp4"> | <video src="https://user-images.githubusercontent.com/32963687/231411858-5f4a7f82-6137-451e-8ed0-1d285618408f.mp4"> | <video src="https://user-images.githubusercontent.com/32963687/231411922-3bf00316-76c9-40da-99fc-c27227acbfec.mp4"> |

&nbsp;
<table style="width:100%">
        <p align="right">Table 2. Ablated area at different temperatures<p>
    <tr>
        <th valign="middle" align="center">T </th>
        <th valign="middle" align="center">Absorbance</th>
        <th valign="middle" align="center">HSV</th>
        <th valign="middle" align="center">Reflectance</th>
    </tr>
    <tr>
        <td valign="middle" align="center">35</td>
        <td valign="middle" align="center"><img src="media/abs_35.png"  alt="Absorbance (start)" width="225"></td>
        <td valign="middle" align="center"><img src="media/hsv_35.png" alt="HSV (start)" width="225"></td>
        <td valign="middle" align="center"><img src="media/ref_35.png" alt="Reflectance (start)" width="225"></td>
    </tr>
    <tr>
        <td valign="middle" align="center">60</td>
        <td valign="middle" align="center"><img src="media/abs_70.png"  alt="Absorbance (mid)" width="225"></td>
        <td valign="middle" align="center"><img src="media/hsv_70.png" alt="HSV (mid)" width="225"></td>
        <td valign="middle" align="center"><img src="media/ref_70.png" alt="Reflectance (mid)" width="225"></td>
    </tr>
    <tr>
        <td valign="middle" align="center">110</td>
        <td valign="middle" align="center"><img src="media/abs_110.png"  alt="Absorbance (end)" width="225"></td>
        <td valign="middle" align="center"><img src="media/hsv_110.png" alt="HSV (end)" width="225"></td>
        <td valign="middle" align="center"><img src="media/ref_110.png" alt="Reflectance (end)" width="225"></td>
    </tr>
</table>


## Data Access

1. To download the data, clone the repository:
``` bash
git clone https://github.com/ViacheslavDanilov/hsi_analysis.git
```

2. Install DVC:
``` bash
pip install dvc[gdrive]==2.41.1
```

3. Download the dataset(s) using the following DVC commands:

|                                                                     Dataset                                                                     |                                                                                       Description                                                                                        | Size, Gb |                Command                 |
|:-----------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------:|:--------------------------------------:|
|             [Raw](https://github.com/ViacheslavDanilov/hsi_analysis/blob/1708d39924faecec882468aa0c2ddff340f4677f/dvc/data/raw.dvc)             |                                                              Dataset based on 26 experiments with 304 hyperspectral images                                                               |   34.4   |    ```dvc pull dvc/data/raw.dvc```     |
|  [Supervisely (input)](https://github.com/ViacheslavDanilov/hsi_analysis/blob/1708d39924faecec882468aa0c2ddff340f4677f/dvc/data/sly_input.dvc)  |                                                    Dataset used for labeling on the [Sueprvsiely](https://supervisely.com/) platform                                                     |   7.5    | ```dvc pull dvc/data/sly_input.dvc```  |
| [Supervisely (output)](https://github.com/ViacheslavDanilov/hsi_analysis/blob/1708d39924faecec882468aa0c2ddff340f4677f/dvc/data/sly_output.dvc) | Dataset that represents the labeled dataset for object detection in a [Supervisely format](https://docs.supervise.ly/data-organization/00_ann_format_navi/04_supervisely_format_objects) |   2.6    | ```dvc pull dvc/data/sly_output.dvc``` |
|         [Interim](https://github.com/ViacheslavDanilov/hsi_analysis/blob/1708d39924faecec882468aa0c2ddff340f4677f/dvc/data/interim.dvc)         |                                                                 Dataset used for debugging and explanatory data analysis                                                                 |   2.6    |  ```dvc pull dvc/data/interim.dvc```   |
|            [COCO](https://github.com/ViacheslavDanilov/hsi_analysis/blob/1708d39924faecec882468aa0c2ddff340f4677f/dvc/data/coco.dvc)            |                                        Dataset in [COCO format](https://cocodataset.org/#format-data) and used to train object recognition models                                        |   10.5   |    ```dvc pull dvc/data/coco.dvc```    |
