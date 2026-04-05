# DA6401 Assignment 2: Multi-Stage Visual Perception Pipeline

This repository implements a full PyTorch perception pipeline on the Oxford-IIIT Pet dataset:

- Breed classification (37 classes)
- Single-object localization (bounding box regression)
- Semantic segmentation (trimap prediction)
- Unified multi-task inference in one forward pass

The project follows the Assignment-2 skeleton structure and required class interfaces.

## 1. Repository Overview

Main training and evaluation scripts:

- [train.py](train.py): Task 1, VGG11 classification training
- [train_localizer.py](train_localizer.py): Task 2, localization training
- [train_unet.py](train_unet.py): Task 3, U-Net style segmentation training
- [test_dropout.py](test_dropout.py), [test_iou.py](test_iou.py), [test_vgg_encoder.py](test_vgg_encoder.py), [test_classifier.py](test_classifier.py), [test_localizer.py](test_localizer.py), [test_unet.py](test_unet.py), [test_multitask.py](test_multitask.py), [test_dataset.py](test_dataset.py): quick functional tests
- [pipeline_demo.py](pipeline_demo.py), [visualize_feature.py](visualize_feature.py), [visualize_detection.py](visualize_detection.py), [visualize_segmentation.py](visualize_segmentation.py): visualization and report helpers

Core modules:

- [models/vgg11.py](models/vgg11.py): VGG11 backbone and classifier model
- [models/layers.py](models/layers.py): CustomDropout implementation
- [losses/iou_loss.py](losses/iou_loss.py): IoU loss for bbox regression
- [models/localization.py](models/localization.py): localization model
- [models/segmentation.py](models/segmentation.py): U-Net style segmentation model
- [models/multitask.py](models/multitask.py): unified multi-task model
- [multitask.py](multitask.py): root-level import shim for autograder compatibility

Data and checkpoints:

- [data/pets_dataset.py](data/pets_dataset.py): dataset loader for image, class label, bbox, trimap mask
- [checkpoints](checkpoints): stores classifier, localizer, and unet weights

## 2. Required Public Interfaces

The following interfaces are available as required:

- from models.vgg11 import VGG11
- from models.layers import CustomDropout
- from losses.iou_loss import IoULoss
- from multitask import MultiTaskPerceptionModel

## 3. Model Design Summary

### 3.1 Task 1: Classification

- VGG11 architecture implemented from scratch in [models/vgg11.py](models/vgg11.py)
- BatchNorm is configurable with use_batchnorm
- CustomDropout is used in the classifier head
- Input image size is fixed to 224 x 224

### 3.2 Task 2: Localization

- Encoder reused from VGG11
- Regression output format: [x_center, y_center, width, height] in pixel space
- Localizer head output is constrained to image coordinates via sigmoid scaling
- Training loss in [train_localizer.py](train_localizer.py): MSE + IoULoss

### 3.3 Task 3: Segmentation

- Encoder: VGG11Encoder
- Decoder: symmetric transposed-convolution decoder with skip concatenations
- Output channels: 3 (trimap classes)
- Training objective in [train_unet.py](train_unet.py): CrossEntropy + multiclass Dice loss

### 3.4 Task 4: Unified Multi-Task

[models/multitask.py](models/multitask.py) builds a shared encoder and three heads:

- Classification logits, shape [B, 37]
- Bounding boxes, shape [B, 4]
- Segmentation logits, shape [B, 3, 224, 224]

The model loads pretrained weights from:

- checkpoints/classifier.pth
- checkpoints/localizer.pth
- checkpoints/unet.pth

## 4. Dataset Handling

The dataset loader in [data/pets_dataset.py](data/pets_dataset.py):

- Reads split files from data/annotations
- Loads and resizes images to 224 x 224
- Resizes trimaps with nearest-neighbor and maps labels to 0, 1, 2
- Parses XML bounding boxes and rescales coordinates to the resized image
- Returns dictionary keys: image, label, bbox, mask

## 5. Environment Setup

### 5.1 Create and activate virtual environment

Windows PowerShell:

```powershell
python -m venv env_gpu
.\env_gpu\Scripts\Activate.ps1
```

### 5.2 Install dependencies

```powershell
pip install -r requirements.txt
```

### 5.3 Optional: login to W&B

```powershell
wandb login
```

## 6. Training Workflow

Run in this order:

```powershell
python .\train.py
python .\train_localizer.py
python .\train_unet.py
```

Notes:

- [train.py](train.py) logs classification runs to W&B and supports dropout experiments by editing dropout_p.
- [train_unet.py](train_unet.py) supports transfer strategy via transfer_type values: freeze, partial, full.
- The best model per training script is saved into [checkpoints](checkpoints).

## 7. Quick Verification

Run all test scripts:

```powershell
python .\test_dropout.py
python .\test_iou.py
python .\test_vgg_encoder.py
python .\test_classifier.py
python .\test_localizer.py
python .\test_unet.py
python .\test_multitask.py
python .\test_dataset.py
```

Expected multitask output shapes:

- classification: torch.Size([2, 37])
- localization: torch.Size([2, 4])
- segmentation: torch.Size([2, 3, 224, 224])

## 8. W&B Report Support (Part 2)

Useful scripts for report artifacts:

- [train.py](train.py): dropout runs and loss/accuracy curves
- [train_unet.py](train_unet.py): transfer strategy comparison (freeze/partial/full)
- [visualize_feature.py](visualize_feature.py): early vs late feature map visualization
- [visualize_detection.py](visualize_detection.py): detection table with bbox overlays and IoU
- [visualize_segmentation.py](visualize_segmentation.py): segmentation table and pixel metrics
- [pipeline_demo.py](pipeline_demo.py): final pipeline showcase on external images

For final showcase images, place files in [test_images](test_images) (or update paths in [pipeline_demo.py](pipeline_demo.py)).

## 9. Checkpoints

Required checkpoint files:

- [checkpoints/classifier.pth](checkpoints/classifier.pth)
- [checkpoints/localizer.pth](checkpoints/localizer.pth)
- [checkpoints/unet.pth](checkpoints/unet.pth)

Checkpoint format used by training scripts:

```python
{
	"state_dict": model.state_dict(),
	"epoch": epoch,
	"best_metric": metric_value,
}
```

## 10. Project Structure

```text
.
|-- checkpoints/
|-- data/
|   |-- annotations/
|   |-- images/
|   `-- pets_dataset.py
|-- losses/
|   `-- iou_loss.py
|-- models/
|   |-- classification.py
|   |-- layers.py
|   |-- localization.py
|   |-- multitask.py
|   |-- segmentation.py
|   `-- vgg11.py
|-- multitask.py
|-- train.py
|-- train_localizer.py
|-- train_unet.py
|-- test_*.py
`-- README.md
```

## 11. Links

- GitHub Repository: [DA6401 Assignment 2](https://github.com/treasureNIDHI/DA6401_assignment2)
- W&B Report: [Assignment-2-Multitask](https://wandb.ai/nidhi-jagatpura-iit-madras/DA6401_A2_Multitask/reports/Assignment-2-Multitask--VmlldzoxNjQxMTc4NQ?accessToken=5t37r0e5m0i7pmvauapaw01pjhlbfgu8h3hth3p0yijv5g680bhjjb9mxulqyld8)

