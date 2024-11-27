# Intelligent-Systems-Course-Project
A comprehensive implementation of 2D/3D U-Net variants (including attention mechanisms and residual connections) for brain tumor segmentation using the BraTS 2020 dataset
# 🧠 Brain Tumor Segmentation Using U-Net Architectures
A comprehensive implementation of 2D/3D U-Net variants (including attention mechanisms and residual connections) for brain tumor segmentation using the BraTS 2020 dataset, developed and tested on NVIDIA A5000 GPU.
# 🖥️ Hardware Requirements
GPU Configuration Used:

NVIDIA A5000 (24GB VRAM)

Important Note:

A GPU with significant VRAM is required to run these models, especially the 3D architectures
Training these models on CPUs is not recommended due to computational intensity
Lower VRAM GPUs may require batch size adjustments or model modifications

# 📥 Prerequisites
Before running the notebooks, you'll need to:

Download the BraTS 2020 dataset from Kaggle:
BraTS 2020 Training Dataset 
[![Dataset](https://img.shields.io/badge/Dataset-BraTS2020-blue)](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data)

Ensure you have a CUDA-compatible GPU with sufficient VRAM (16GB+ recommended)

# 🛠️ Installation

Create and activate a conda environment:
```
conda create -n brain_seg python=3.8
conda activate brain_seg
```
Install required packages:
```
pip install -r requirements.txt
```

Required packages include:
```
segmentation-models-pytorch
matplotlib
Pillow
numpy
torch
torchmetrics
torchvision
albumentations
pandas
nibabel
tqdm

```

# 📁 Repository Structure
The project is divided into multiple Jupyter notebooks, each focusing on a specific architecture:

UNet_2D.ipynb - Implementation of traditional 2D U-Net
Attention_ResUNet.ipynb - Enhanced U-Net with residual connections and attention mechanism
Ensemble-Attention_ResUNet.ipynb - Mixture of experts approach using Attention ResU-Net
UNet_3D.ipynb - Three-dimensional U-Net implementation
ResUNet_3D.ipynb - 3D U-Net with residual connections
Attention_ResUNet_3D.ipynb - 3D architecture combining attention mechanisms with residual connections

# 🔬 Key Findings
Our experiments demonstrate that 3D architectures consistently outperform their 2D counterparts, leveraging the volumetric nature of MRI scans. The ensemble approach using Attention ResU-Net showed particularly promising results, highlighting the benefits of combining multiple expert models.
The implementations provide a solid foundation for researchers and practitioners working on medical image segmentation tasks, especially those focused on brain tumor detection and segmentation.
Note: Each notebook is self-contained and can be run independently after setting up the dataset and ensuring appropriate GPU resources are available.
