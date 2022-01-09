# BrainTumorDetection
Image Classification and Segmentation for Brain Tumor Detection

### Ipynb files were run on Google Colab.
Classification dataset : https://www.kaggle.com/masoudnickparvar/brain-tumor-mri-dataset
Segmentation dataset : https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation

## Image Classification model with EfficientNetB3(With ImageNet weights)
The model's json and h5 files are stored as "model_classification.json" and "model_classification.h5"

 Train Accuracy  | Test Accuracy | Micro-Average AUROC| 
| ------------- | ------------- | ------------- |
| 99.98%  | 99.39%  | 0.996




## Image Segmentation model with UNet-VGG16
The model's json and h5 files are stored as "model_segmentation.json" and "model_segmentation.h5".
The results below are gathered after running 50 epochs but we did not store the model. Hence, the current json and h5 file for segmentation are retrieved after running 15 epochs only.

Test IOU  | Test DICE | 
| ------------- | ------------- | 
| 0.8918  | 0.8996  |

## How the model works?
![](figures/FlowChat.PNG)

## How to run the code?
Keep all the h5 files and main.py in a folder. CD to that folder and run main.py on command prompt. 

## Errors & Improvements
