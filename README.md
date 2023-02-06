# Image-classification-3d
Radiogenomic image classification of MGMT promoter methylation status in glioblastoma patients

# Design
This repository contains code to create a baseline model for 3D image classification using MRI scans. The scripts can be used with several command line arguments to perform relevant experiments. Multiple performance metrics, as well as the model's best train weight, will be available in the **Results** folder.

## Dataset 
Please download the dataset from the Kaggle challenge website **RSNA-MICCAI Brain Tumor Radiogenomic Classification**

## Usage
### Major packages required
* Pytorch
* scikit-learn
* Monai

### Directory structure
* code directory (example: exp1)
* Input/raw/exp1
* Input/prep/exp1
* Output/exp1/Results/
### Prefil config file before running experiments
* Use create_folds.py to split the dataset file into 5 folds.
* To obtain the baseline, run train.py with command-line arguments.
* Execute train_loop.sh to extend the preceding result to all data folds (five in this case).
