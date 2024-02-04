# NAME
The official codes for [**Knowledge-enhanced Visual-Language Pre-training on Chest Radiology Images**](https://arxiv.org/pdf/2302.14042.pdf).

## Dependencies

To clone all files:

```
git clone 
```

To install Python dependencies:

```
pip install -r requirements.txt
```

**Note that the complete data file and model training logs/checkpoints can be download from link :https://pan.baidu.com/s/1qFRuJNNmcL0pC1AtjFjbaw  (abrz). and google drive with link: [https://drive.google.com/drive/folders/1xWVVJRfnm_wIgUpbn9ftsW4K5XKU0i0-?usp=share_link](https://drive.google.com/drive/folders/1xWVVJRfnm_wIgUpbn9ftsW4K5XKU0i0-?usp=sharing).**


## Data

#### **Training Dataset**   
**1. MIMIC-CXR Dataset**

Navigate to [MIMIC-CXR Database](https://physionet.org/content/mimic-cxr/2.0.0/) to download the training dataset. Note: in order to gain access to the data, you must be a credentialed user as defined on [PhysioNet](https://physionet.org/settings/credentialing/).

**2. CheXpert Dataset**

The CheXpert dataset consists of chest radiographic examinations from Stanford Hospital, performed between October 2002 and July 2017 in both inpatient and outpatient centers. Population-level characteristics are unavailable for the CheXpert test dataset, as they are used for official evaluation on the CheXpert leaderboard.

The main data (CheXpert data) supporting the results of this study are available at https://aimi.stanford.edu/chexpert-chest-x-rays.


#### **Evaluation Dataset**   

**1. CheXpert Dataset**

The CheXpert **test** dataset has recently been made public, and can be found by following the steps in the [cheXpert-test-set-labels](https://github.com/rajpurkarlab/cheXpert-test-set-labels) repository. 

**2. PadChest Dataset**

The PadChest dataset contains chest X-rays that were interpreted by 18 radiologists at the Hospital Universitario de San Juan, Alicante, Spain, from January 2009 to December 2017. The dataset contains 109,931 image studies and 168,861 images. PadChest also contains 206,222 study reports.

The [PadChest](https://arxiv.org/abs/1901.07441) is publicly available at https://bimcv.cipf.es/bimcv-projects/padchest. Those who would like to use PadChest for experimentation should request access to PadChest at the [link](https://bimcv.cipf.es/bimcv-projects/padchest).

**3. ChestX-Det-10 dataset**




## Prepare Data and Weights 

Download three files: pretrained_mimic_diffusion, mimic_models and paired data from https://pan.baidu.com/s/1qFRuJNNmcL0pC1AtjFjbaw  (abrz) or: [https://drive.google.com/drive/folders/1xWVVJRfnm_wIgUpbn9ftsW4K5XKU0i0-?usp=share_link](https://drive.google.com/drive/folders/1xWVVJRfnm_wIgUpbn9ftsW4K5XKU0i0-?usp=sharing).**
Put them into our root dir.


## Training and testing

**1. Stage1**

run the following command to perform stage1 training on ControlNet to train a rib constraint generative model
`python stage1_train.py ` 

run the following command to perform stage1 testing on ControlNet to generate paired normal/diseased data
`python stage1_generate_paired_image.py ` 

run the following command to perform stage1 post-precessing paired normal/diseased data
`python minus.py ` 
`python gray2hotmap.py ` 


**2. Stage2**
run the following command to perform stage2 training on Transunet(trained on train split of Generated paired data) to train a pathology localization and segmentation model
`python stage2_train.py ` 

run the following command to perform stage2 testing on Transunet(inference on test split of Generated paired data) to test a pathology localization and segmentation model
`python stage2_test_generateddata.py ` 

run the following command to perform stage2 testing on Transunet(inference on test split of CheXpert dataset) to test a pathology localization and segmentation model
`python stage2_test_chexpert.py ` 

run the following command to perform stage2 testing on Transunet(inference on test split of ChestX-Det-10 dataset) to test a pathology localization and segmentation model
`python stage2_test_chestdet.py ` 

run the following command to perform stage2 testing on Transunet(inference on test split of PadChest dataset) to test a pathology localization and segmentation model
`python stage2_test_padchest.py ` 




If you have any question, please feel free to contact.








