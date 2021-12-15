# BrainTumorDetection
Image Classification and Segmentation for Brain Tumor Detection

## Image Classification model with EfficientNetB3(With ImageNet weights)
The model's json and h5 files are stored as "model_classification.json" and "model_classification.h5"

 Train Accuracy  | Test Accuracy | Micro-Average AUROC| 
| ------------- | ------------- | ------------- |
| 99.98%  | 99.39%  | 0.996




## Image Segmentation model with UNet-VGG16
The model's json and h5 files are stored as "model_segmentation.json" and "model_segmentation.h5".
The results below are gathered after running 50 epochs but we did not store the model. Hence, the current json and h5 file for segmentation are retrieved after running 20 epochs.

Test IOU  | Test DICE | 
| ------------- | ------------- | 
| 0.8918  | 0.8996  |
