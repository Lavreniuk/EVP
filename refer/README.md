# Referring Image Segmentation
## Getting Started 

1. Install the required packages.

```
pip install -r requirements.txt
```

2. Prepare RefCOCO datasets following [LAVT](https://github.com/yz93/LAVT-RIS).

* Download COCO 2014 Train Images [83K/13GB] from [COCO](https://cocodataset.org/#download), and extract `train2014.zip` to `./refer/data/images/mscoco/images`

* Follow the instructions in `./refer` to download and extract `refclef.zip, refcoco.zip, refcoco+.zip, refcocog.zip` to `./refer/data`

Your dataset directory should be:

```
refer/
├──data/
│  ├── images/mscoco/images/
│  ├── refclef
│  ├── refcoco
│  ├── refcoco+
│  ├── refcocog
├──evaluation/
├──...
```

## Results and Fine-tuned Models of EVP
EVP achieves 76.35 overall IoU and 77.61 mean IoU on the validation set of RefCOCO.

## Training

We count the max length of referring sentences and set the token length of lenguage model accrodingly. The checkpoint of the best epoch would be saved at `./checkpoints/`.

* Train on RefCOCO

```
bash train.sh refcoco /path/to/logdir <NUM_GPUS> --token_length 40
```

* Train on RefCOCO+

```
bash train.sh refcoco+ /path/to/logdir <NUM_GPUS> --token_length 40
```

* Train on RefCOCOg

```
bash train.sh refcocog /path/to/logdir <NUM_GPUS> --token_length 77 --splitBy umd
```

## Evaluation

* Evaluate on RefCOCO

```
bash test.sh refcoco /path/to/evp_ris_refcoco.pth --token_length 40
```

* Evaluate on RefCOCO+

```
bash test.sh refcoco+ /path/to/evp_ris_refcoco+.pth --token_length 40
```

* Evaluate on RefCOCOg

```
bash test.sh refcocog /path/to/evp_ris_gref.pth --token_length 77 --splitBy umd
```

## Custom inference
```
PYTHONPATH="../":$PYTHONPATH python inference.py --img_path test_img.jpg --resume refcoco.pth --token_length 40 --prompt 'green plant'
```
