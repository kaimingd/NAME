# PIXEL
The official codes for **Prompt-driven Healthy/Diseased Image Pairs Enabling Pixel-level Chest X-ray Pathology Localization**.

## Dependencies

To clone all files:

```
git clone git@github.com:kaimingd/PIXEL.git
```

To install Python dependencies:

conda create -n pixel python=3.8

```
conda create -n pixel python=3.8
conda activate pixel
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

**Note that the complete data file and model training logs/checkpoints can be download from link : https://pan.baidu.com/s/1oPoYeFsia3ngIsrSEWgX_w?pwd=refg (refg).**


## Data

#### **Training Dataset**   
**1. MIMIC-CXR Dataset**

Navigate to [MIMIC-CXR Database](https://physionet.org/content/mimic-cxr/2.0.0/) to download the training dataset. Note: in order to gain access to the data, you must be a credentialed user as defined on [PhysioNet](https://physionet.org/settings/credentialing/).

**2. CheXpert Dataset**

The CheXpert dataset consists of chest radiographic examinations from Stanford Hospital, performed between October 2002 and July 2017 in both inpatient and outpatient centers. Population-level characteristics are unavailable for the CheXpert test dataset, as they are used for official evaluation on the CheXpert leaderboard.

The main data (CheXpert data) supporting the results of this study are available at https://aimi.stanford.edu/chexpert-chest-x-rays.


#### **Evaluation Dataset**   

**1. CheXpert Dataset**

The CheXpert test dataset has recently been made public, and can be found by following the steps in the [cheXpert-test-set-labels](https://github.com/rajpurkarlab/cheXpert-test-set-labels) repository. 

**2. PadChest Dataset**

The PadChest dataset contains chest X-rays that were interpreted by 18 radiologists at the Hospital Universitario de San Juan, Alicante, Spain, from January 2009 to December 2017. The dataset contains 109,931 image studies and 168,861 images. PadChest also contains 206,222 study reports.

The [PadChest](https://arxiv.org/abs/1901.07441) is publicly available at https://bimcv.cipf.es/bimcv-projects/padchest. Those who would like to use PadChest for experimentation should request access to PadChest at the [link](https://bimcv.cipf.es/bimcv-projects/padchest).

**3. ChestX-Det-10 dataset**

ChestX-Det10 is a subset with instance-level box annotations of [NIH ChestX-14](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community).
For image downloading, please visit http://resource.deepwise.com/xraychallenge/train_data.zip and http://resource.deepwise.com/xraychallenge/test_data.zip.

## Prepare Data and Weights 

Download five files: pretrained_mimic_diffusion, mimic_models, paired data, images_transunet_minus1/unet-6v-latest.pt, images_transunet_padchest_minus1/unet-6v-latest.pt
from  https://pan.baidu.com/s/1oPoYeFsia3ngIsrSEWgX_w?pwd=refg (refg).
Put them into our root dir.


## Training and testing

**1. Stage1**

Run the following command to perform stage1 training on ControlNet to train a rib constraint generative model

`python stage1_train.py ` 

Run the following command to perform stage1 testing on ControlNet to generate paired normal/diseased data

`python stage1_generate_paired_image.py ` 

Run the following command to perform stage1 post-precessing paired normal/diseased data

`python minus.py ` 

`python gray2hotmap.py ` 


**2. Stage2**

Run the following command to perform stage2 training on Transunet(trained on train split of Generated paired data) to train a pathology localization and segmentation model

`python stage2_train.py ` 

Run the following command to perform stage2 testing on Transunet(inference on test split of Generated paired data) to test a pathology localization and segmentation model

`python stage2_test_generateddata.py ` 

Run the following command to perform stage2 testing on Transunet(inference on test split of CheXpert dataset) to test a pathology localization and segmentation model

`python stage2_test_chexpert.py ` 

Run the following command to perform stage2 testing on Transunet(inference on test split of ChestX-Det-10 dataset) to test a pathology localization and segmentation model

`python stage2_test_chestdet.py ` 

Run the following command to perform stage2 testing on Transunet(inference on test split of PadChest dataset) to test a pathology localization and segmentation model

`python stage2_test_padchest.py ` 




If you have any question, please feel free to contact.








