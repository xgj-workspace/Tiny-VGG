
# TINY-VGG IMAGE CLASSIFICATION PIPELINE (PYTORCH)


A lightweight PyTorch implementation of the TinyVGG Convolutional Neural 
Network architecture for computer vision and image classification. Inspired 
by the CNN Explainer (https://poloclub.github.io/cnn-explainer/) model 
layout, this implementation adds modern training enhancements including 
BatchNorm2d, Dropout, dynamic learning rate decay (ReduceLROnPlateau), and 
an Early Stopping mechanism for superior training stability and validation 
performance.

It is for machine learning beginer and serves as a starting point 
for Convolutional Neural Networks (CNNs).

The pipeline offers out-of-the-box hardware acceleration for Apple Silicon 
GPUs (MPS), NVIDIA GPUs (CUDA), and CPU fallbacks.

---
## 1. KEY FEATURES


- Upgraded Architecture: Incorporates BatchNorm2d in convolutional blocks 
  and Dropout(p=0.5) in the classifier head to prevent overfitting.
- Cross-Platform Acceleration: Native support for Apple Silicon hardware 
  acceleration via PyTorch's MPS backend, complete with automated environment 
  variable fallback overrides.
- Dual Dataset Pipeline:
  * CIFAR-10 Default: Trains immediately without needing local image preparation.
  * Custom Dataset Directory: Easily switch to custom datasets using standard 
    ImageFolder directory structures.
- Robust Training Pipeline:
  * Early Stopping: Tracks validation loss and automatically saves the best-
    performing model checkpoint ('best_tinyvgg_model.pth').
  * LR Scheduler: Reduces learning rate dynamically when validation loss plateaus.
  * Data Augmentation: Includes spatial transforms (RandomHorizontalFlip, 
    RandomRotation) and color jitters.
- CLI Inference Tool: Visual prediction script that renders class probability 
  distributions with ASCII progress bars.

--- 
## 2. PROJECT STRUCTURE

```
.
├── tiny_VGG.py              # Main training and validation script
├── test_model.py            # Single-image inference script
├── best_tinyvgg_model.pth   # Model weights (generated upon training)
└── dataset/                 # (Optional) Custom dataset directory
    ├── train/
    │   ├── class_a/
    │   └── class_b/
    └── val/
        ├── class_a/
        └── class_b/

```

---
## 3. PREREQUISITES & INSTALLATION
```

Ensure you have Python 3.8 or higher installed along with PyTorch, 
Torchvision, and Pillow:

  pip install torch torchvision pillow

```
---
## 4. USAGE GUIDE

```
[A] TRAINING (tiny_VGG.py)

Option 1: Train on CIFAR-10 (Default)
--------------------------------------
No local images required. PyTorch will automatically download CIFAR-10 into a 
'data/' directory:

  python tiny_VGG.py  


Option 2: Train on a Custom Dataset
-----------------------------------
Structure your dataset into 'dataset/train/<class_name>/' and 
'dataset/val/<class_name>/', then pass the '--custom-dataset' flag:

  python tiny_VGG.py --custom-dataset  



[B] SINGLE IMAGE INFERENCE (test_model.py)

Once training completes, the pipeline saves the optimal weights to 
'best_tinyvgg_model.pth'.

Infer using CIFAR-10 Labels (Default):
--------------------------------------
  python test_model.py path/to/target_image.jpg


Infer using Custom Dataset Labels:
----------------------------------
If your checkpoint was trained using custom local classes, append 
'--custom-dataset':

  python test_model.py path/to/target_image.jpg --custom-dataset


Sample Terminal Output:
-----------------------
=======================================================
🔮 INFERENCE RESULT
=======================================================
🖼️ Target Image:     sample_car.jpg
🏆 Top Prediction:   AUTOMOBILE (95.2% confidence)
-------------------------------------------------------
📊 Probability Breakdown:
  • airplane    :   0.3% | 
  • automobile  :  95.2% | ███████████████████
  • bird        :   0.1% | 
  • cat         :   0.4% | 
  • deer        :   0.1% | 
  • dog         :   0.2% | 
  • frog        :   0.1% | 
  • horse       :   0.1% | 
  • ship        :   1.2% | 
  • truck       :   2.3% | 
=======================================================

```
---
## 5. MODEL ARCHITECTURE SUMMARY 
```
Input Tensor (3 x 64 x 64)
  │
  ├── [Conv2d(3 -> 16, k=3, p=1)  -> BatchNorm2d -> ReLU]
  ├── [Conv2d(16 -> 16, k=3, p=1) -> BatchNorm2d -> ReLU]
  ├── MaxPool2d(kernel_size=2)   --> Feature Map: (16 x 32 x 32)
  │
  ├── [Conv2d(16 -> 16, k=3, p=1) -> BatchNorm2d -> ReLU]
  ├── [Conv2d(16 -> 16, k=3, p=1) -> BatchNorm2d -> ReLU]
  ├── MaxPool2d(kernel_size=2)   --> Feature Map: (16 x 16 x 16)
  │
  ├── Flatten Layer
  ├── Dropout(p=0.5)
  └── Linear(in_features=4096, out_features=num_classes)

``` 
---
## 6. LICENSE 

This project is licensed under the MIT License. Feel free to modify, 
distribute, or fork for your own machine learning research! 
