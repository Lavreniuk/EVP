# Depth Estimation
## Getting Started

1. Install the [mmcv-full](https://github.com/open-mmlab/mmcv) library and some required packages.

```bash
pip install openmim
mim install mmcv-full
pip install -r requirements.txt
```

2. Prepare NYUDepthV2 datasets following [GLPDepth](https://github.com/vinvino02/GLPDepth) and [BTS](https://github.com/cleinc/bts/tree/master).

```
mkdir nyu_depth_v2
wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
python extract_official_train_test_set_from_mat.py nyu_depth_v2_labeled.mat splits.mat ./nyu_depth_v2/official_splits/
```

Download sync.zip provided by the authors of BTS from this [url](https://drive.google.com/file/d/1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP/view) and unzip in `./nyu_depth_v2` folder. 

Your dataset directory should be:

```
│nyu_depth_v2/
├──official_splits/
│  ├── test
│  ├── train
├──sync/
```

## Results and Fine-tuned Models

EVP obtains 0.224 RMSE on NYUv2 depth estimation benchmark, establishing the new state-of-the-art.

|  | RMSE | d1 | d2 | d3 | REL  | log_10 |
|---------|-------|-------|--------|------|-------|-------|
| **EVP** | 0.224 | 0.976 | 0.997 | 0.999 | 0.061 | 0.027 |

EVP obtains 0.048 REL and 0.136 SqREL on KITTI depth estimation benchmark, establishing the new state-of-the-art.

|  | REL | SqREL | RMSE | RMSE log | d1 | d2 | d3 |
|---------|-------|-------|--------|------|-------|-------|-------|
| **EVP** | 0.048 | 0.136 | 2.015 | 0.073 | 0.980 | 0.998 | 1.000 |

## Training

Run the following instuction to train the EVP-Depth model.

```
bash train.sh <LOG_DIR>
```

## Evaluation
Command format:
```
bash test.sh <CHECKPOINT_PATH>
```

## Custom inference
```
PYTHONPATH="../":$PYTHONPATH python inference.py --img_path test_img.jpg --ckpt_dir nyu.ckpt
```
