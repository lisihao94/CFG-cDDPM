
# CFG-cDDPM

This is part of the original data set and test results of our research. The manuscript of the paper is being revised and will be indexed after its official publication.

The files in **trainingdata** are partial examples; you can create more similar examples based on your own data and requirements. 

For the specific data‑processing methods used in this study, please refer to the paper. 

Please set parameters according to your data volume and equipment.

The python library and version used in this study are in **requirements.txt** The official library may be updated, resulting in code errors. Please select the appropriate library to install according to the official version.

After completing data processing, code debugging and environment configuration, run **train.py** to start your training.

Here is the accurate direct translation into English, maintaining your original Markdown formatting and code block structures:



## Quick Start
1. Create a local project root directory `YOUR_PROJECT_ROOT`, and change `root_dir` in `configs/config.yaml` to this absolute path.

2. Prepare your data:
- Place original training images in `YOUR_PROJECT_ROOT/data/origin_train/`, and place `train_meta.csv` in the same directory, format:


********************************************
train:
python train.py ..\configs\config.yaml
********************************************
test:
python test.py ..\configs\config.yaml

## Mask Mode Configurations (Configurable separately for train/val/test)

- In the `masks` section of `configs/config.yaml`, select different mask modes independently for the training, validation, and testing stages:



masks:
train:
mode: "hybrid"           # center | random | hybrid | full | self
center_scale: 0.5         # center/hybrid: width & height ratio of the center rectangle
random_ratio: 0.3         # random: random mask area ratio (0~1)
hybrid_random_ratio: 0.2  # hybrid: additional random mask area ratio (0~1)
self_mask_dir: "${root_dir}/data/train/masks"  # self: custom mask directory (optional)
val:
mode: "center"
center_scale: 0.5
random_ratio: 0.5
hybrid_random_ratio: 0.25
self_mask_dir: "${root_dir}/data/val/masks"
test:
mode: "self"              # Default reads masks with identical filenames from `${root_dir}/data/test/masks`
center_scale: 0.5
random_ratio: 0.5
hybrid_random_ratio: 0.25
self_mask_dir: "${root_dir}/data/test/masks"

```

- Mode Description:
  - `center mask`: A rectangular mask centered on the image, with width and height equal to `center_scale` of the image;
  - `random mask`: Generates continuous "cloud-like" patch masks according to `random_ratio` (by smoothing a random field and selecting via quantile thresholding; masked area roughly equals `random_ratio`, non-global scatter points);
  - `hybrid mask`: Superimposes an additional `hybrid_random_ratio` random mask on top of the central rectangle;
  - `full mask`: Uses the entire image as the mask;
  - `self mask`: Reads masks with identical filenames as the images from a specified directory (black pixels represent masked regions).

- If using `self` mode during training and validation stages, ensure that the corresponding mask directory exists and contains mask files with identical names as the samples; the testing stage defaults to using `data/test/masks`.

## Progressive Masks

- Make the mask linearly increase/decrease over training epochs (e.g., center increases from 0.5 to 1.0, random increases from 0.1 to 1.0). Simply set `*_start` and `*_end` in the `masks` section of the corresponding stage:


```

masks:
train:
mode: "center"
center_scale_start: 0.5
center_scale_end: 1.0
val:
mode: "random"
random_ratio_start: 0.1
random_ratio_end: 1.0
test:
mode: "hybrid"
center_scale: 0.5               # Uses fixed value if *_start/_end are not set
hybrid_random_ratio_start: 0.1
hybrid_random_ratio_end: 0.5

```

  - The validation stage also supports the same progressive configuration (remains fixed if not set).
  - `self` mode does not participate in area progression (determined by external masks).

## Mask Parameter Schedules

- Choose a sub-mode under `masks.<stage>.schedule` and provide parameters under `schedule.params`.
  - Supported types: `constant`, `linear`, `cosine`, `exponential`, `step`
  - Parsing priority: `schedule` > `param_schedules` > `*_start/_end`


```

masks:
train:
mode: "random"
schedule:
type: linear           # constant | linear | cosine | exponential | step
params:
random_ratio:
start: 0.5
end: 1.0
center_scale:
value: 0.5
hybrid_random_ratio:
start: 0.1
end: 0.5

val:
mode: "hybrid"
schedule:
type: step
params:
center_scale:
start: 0.4
end: 0.8
n_steps: 4
hybrid_random_ratio:
start: 0.1
end: 0.5

```

- Explanation:
  - `constant`: Always uses `value`.
  - `linear`: Linear interpolation `start -> end`.
  - `cosine`: Cosine ease-in/ease-out interpolation, `w = 0.5*(1 - cos(pi*progress))`.
  - `exponential`: Geometric interpolation `value = start * (end/start)^progress` (requires `start,end > 0`; falls back to linear otherwise).
  - `step`: Piecewise step interpolation with `n_steps` equal steps.

```


**************************************************************************
requirements：
Package                   Version
------------------------- ------------
absl-py                   2.3.1
affine                    2.4.0
asttokens                 3.0.0
attrs                     25.3.0
backcall                  0.2.0
beautifulsoup4            4.14.3
blinker                   1.8.2
cachetools                5.5.2
certifi                   2024.2.2
charset-normalizer        3.3.2
click                     8.1.8
click-plugins             1.1.1.2
cligj                     0.7.2
colorama                  0.4.6
comm                      0.2.3
config                    0.5.1
ConfigArgParse            1.7.1
contourpy                 1.1.1
cycler                    0.12.1
dash                      3.2.0
decorator                 5.2.1
deep-translator           1.11.4
et_xmlfile                2.0.0
executing                 2.2.1
fastjsonschema            2.21.2
filelock                  3.13.4
fiona                     1.10.1
Flask                     3.0.3
fonttools                 4.57.0
fsspec                    2024.3.1
geopandas                 0.13.2
google-auth               2.43.0
google-auth-oauthlib      1.0.0
grpcio                    1.70.0
idna                      3.7
imageio                   2.35.1
importlib_metadata        8.5.0
importlib_resources       6.4.5
ipython                   8.12.3
ipywidgets                8.1.7
itsdangerous              2.2.0
jedi                      0.19.2
Jinja2                    3.1.3
jsonschema                4.23.0
jsonschema-specifications 2023.12.1
jupyter_core              5.8.1
jupyterlab_widgets        3.0.15
kiwisolver                1.4.7
lazy_loader               0.4
lpips                     0.1.4
lxml                      6.0.2
Markdown                  3.7
MarkupSafe                2.1.5
matplotlib                3.7.5
matplotlib-inline         0.1.7
mpmath                    1.3.0
narwhals                  1.42.1
nbformat                  5.10.4
nest-asyncio              1.6.0
networkx                  3.1
numpy                     1.24.4
oauthlib                  3.3.1
open3d                    0.19.0
opencv-python             4.10.0.84
openpyxl                  3.1.5
osmnx                     1.9.4
packaging                 25.0
pandas                    2.0.3
parso                     0.8.5
pbr                       0.11.1
pickleshare               0.7.5
pillow                    10.3.0
pip                       23.3.1
pkgutil_resolve_name      1.3.10
platformdirs              4.3.6
plotly                    6.3.0
prompt_toolkit            3.0.52
protobuf                  5.29.5
pure_eval                 0.2.3
pyasn1                    0.6.1
pyasn1_modules            0.4.2
Pygments                  2.19.2
pyogrio                   0.9.0
pyparsing                 3.1.4
pyproj                    3.5.0
python-dateutil           2.9.0.post0
pytorch-fid               0.3.0
pytz                      2025.2
PyWavelets                1.4.1
pywin32                   311
PyYAML                    6.0.3
rasterio                  1.3.11
referencing               0.35.1
requests                  2.31.0
requests-oauthlib         2.0.0
retrying                  1.4.2
rpds-py                   0.20.1
rsa                       4.9.1
scikit-image              0.21.0
scipy                     1.10.1
seaborn                   0.13.2
setuptools                68.2.2
shapely                   2.0.7
six                       1.17.0
snuggs                    1.4.7
soupsieve                 2.7
splitter                  0.1.1
stack-data                0.6.3
sympy                     1.12
tensorboard               2.14.0
tensorboard-data-server   0.7.2
tifffile                  2023.7.10
torch                     2.0.1+cu118
torchvision               0.15.2+cu118
tqdm                      4.66.2
traitlets                 5.14.3
trimesh                   4.8.2
typing_extensions         4.11.0
tzdata                    2025.2
urllib3                   2.2.1
wcwidth                   0.2.13
Werkzeug                  3.0.6
wheel                     0.41.2
widgetsnbextension        4.0.14
zipp                      3.20.2



***************************************************************************
hardware:
The hardware configuration for this experiment included a DELL R730 2U rack-mounted server with 8 × 16 GB memory modules and three NVIDIA T4 GPUs. The NVIDIA-SMI version used was 510.47.03, and the CUDA version is 11.8. The software environment consists of the Linux operating system, Visual Studio Code (version 1.71.0), Anaconda Navigator 2.1.1, and Python 3.8 as the development environment.
