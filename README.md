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

<table style="width:100%">
    <caption>
        Table 1. Example of hyperspectral images taken at different temperatures
    </caption>
    <tr>
        <th valign="middle" align="center">Stage </th>
        <th valign="middle" align="center">Absorbance</th>
        <th valign="middle" align="center">HSV</th>
        <th valign="middle" align="center">Reflectance</th>
    </tr>
    <tr>
        <td valign="middle" align="center">Start</td>
        <td valign="middle" align="center"><img src="media/abs_start.png"  alt="Absorbance (start)" width="225"></td>
        <td valign="middle" align="center"><img src="media/hsv_start.png" alt="HSV (start)" width="225"></td>
        <td valign="middle" align="center"><img src="media/ref_start.png" alt="Reflectance (start)" width="225"></td>
    </tr>
    <tr>
        <td valign="middle" align="center">Middle</td>
        <td valign="middle" align="center"><img src="media/abs_mid.png"  alt="Absorbance (mid)" width="225"></td>
        <td valign="middle" align="center"><img src="media/hsv_mid.png" alt="HSV (mid)" width="225"></td>
        <td valign="middle" align="center"><img src="media/ref_mid.png" alt="Reflectance (mid)" width="225"></td>
    </tr>
    <tr>
        <td valign="middle" align="center">End</td>
        <td valign="middle" align="center"><img src="media/abs_end.png"  alt="Absorbance (end)" width="225"></td>
        <td valign="middle" align="center"><img src="media/hsv_end.png" alt="HSV (end)" width="225"></td>
        <td valign="middle" align="center"><img src="media/ref_end.png" alt="Reflectance (end)" width="225"></td>
    </tr>
</table>
