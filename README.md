# Binary Change Detection — EO-SAR Image Pairs 

## Project Description

This project implements a multimodal deep learning pipeline for EO-SAR change detection using Optical (EO) and Synthetic Aperture Radar (SAR) imagery.

The objective is to identify changed regions between EO pre-event images and SAR post-event images using a segmentation-based deep learning approach.

The project includes:

- EO + SAR multimodal fusion
- Attention-based UNet architecture
- Balanced sampling for class imbalance handling
- Focal Dice Loss for segmentation optimization
- Training, validation, inference, and visualization pipeline

The final model predicts binary change masks indicating changed and unchanged regions.

---

# Model Architecture

The implemented architecture includes:

- EO Encoder Branch
- SAR Encoder Branch
- Attention Fusion Module
- UNet-style Decoder
- Segmentation Head

Key features:

- Multimodal EO-SAR fusion
- Attention-guided feature integration
- Binary segmentation output
- Mixed precision GPU training

---

# Requirements
```
pip install -r requirements.txt
```
```
torch==2.11.0
torchvision==0.26.0
numpy==2.4.4
opencv-python==4.13.0.92
albumentations==2.0.8
matplotlib==3.10.9
tqdm==4.67.3
pyyaml==6.0.3
rasterio==1.5.0
scikit-learn==1.7.1
timm==1.0.27
huggingface_hub==1.14.0

```
## Python Version
```
Python 3.13
```

# Environment Setup
```
python -m venv .venv
.venv\Scripts\activate
```

## Installation of pytorch
CUDA 12.8
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
# Dataset Structure
```
datasets/
│
├── train/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
│
├── val/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
│
└── test/
    ├── pre-event/
    ├── post-event/
    └── target/
```

# Training
```
python train.py
```

# Evaluation
```
python inference.py
```

# Model Weights
```
https://drive.google.com/file/d/1QNoWkjN3p9VLQ3A64TOGMSOtOFPoxDC9/view?usp=sharing
```


# Results 
## Validation
| Metric    | Score  |
| --------- | ------ |
| IoU       | 0.1038 |
| Precision | 0.1379 |
| Recall    | 0.1831 |
| F1        | 0.1386 |


## Test
| Metric    | Score  |
| --------- | ------ |
| IoU       | 0.0318 |
| Precision | 0.0774 |
| Recall    | 0.1056 |
| F1        | 0.0544 |

The model shows meaningful EO-SAR multimodal change localization, though generalization remains challenging due to sparse change regions and modality differences between EO and SAR imagery.



# Citation / References 
```
[1]S. Saha, M. Shahzad, P. Ebel and X. X. Zhu, "Supervised Change Detection Using Prechange Optical-SAR and Postchange SAR Data," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 15, pp. 8170-8178, 2022, doi: 10.1109/JSTARS.2022.3206898.
keywords: {Optical sensors;Optical imaging;Adaptive optics;Synthetic aperture radar;Radar polarimetry;Optical fiber networks;Feature extraction;Change detection (CD);fusion;multisensor analysis;optical images;Siamese network;synthetic aperture radar (SAR)},

[2]Y. Chen and L. Bruzzone, "Self-Supervised Change Detection by Fusing SAR and Optical Multi-Temporal Images," 2021 IEEE International Geoscience and Remote Sensing Symposium IGARSS, Brussels, Belgium, 2021, pp. 3101-3104, doi: 10.1109/IGARSS47720.2021.9553542. keywords: {Change detection algorithms;Geoscience and remote sensing;Optical imaging;Feature extraction;Adaptive optics;Optical sensors;Synthetic aperture radar;Change Detection;Data Fusion;Self-supervised Learning;Sentinel-1/-2;Remote Sensing},

[3]S. Cui, G. Schwarz and M. Datcu, "A Benchmark Evaluation of Similarity Measures for Multitemporal SAR Image Change Detection," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 9, no. 3, pp. 1101-1118, March 2016, doi: 10.1109/JSTARS.2015.2486038.
keywords: {Synthetic aperture radar;Benchmark testing;Mutual information;Indexes;Earth;Random variables;Gaussian distribution;Change detection;change simulation;information similarity measure;synthetic aperture radar (SAR);Change detection;change simulation;information similarity measure;synthetic aperture radar (SAR)},

[4]Wei, J., Zhang, Y., Wu, H., & Cui, B. (2020). An Efficient Change Detection for Large SAR Images Based on Modified U-Net Framework. Canadian Journal of Remote Sensing, 46(3), 272–294. https://doi.org/10.1080/07038992.2020.1783993
```

[5]M. -E. Pegia, B. Þ. Jónsson, A. Moumtzidou, I. Gialampoukidis, S. Vrochidis and I. Kompatsiaris, "Comparative Analysis of Learning-Based Approaches for Change Detection in Satellite Images," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 18, pp. 3766-3781, 2025, doi: 10.1109/JSTARS.2024.3522350.
keywords: {Image resolution;Noise;Training;Remote sensing;Feature extraction;Earth;Bayes methods;Accuracy;Transformers;Roads;Change detection;deep learning;satellite data},

