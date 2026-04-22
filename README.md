# Flowers102 Fine-Grain Recognition

A comparative study of custom CNNs, Vision Transformers (ViT), and Transfer Learning for fine-grain flower recognition.

## Objective
The primary objective is to evaluate the performance of different deep learning architectures on small, fine-grained image datasets. We compare the generalization ability of models trained from scratch (CNNs and ViTs) versus the performance of pre-trained models using **Transfer Learning**.

## Problem it Solves
Small datasets often lead to severe overfitting with deep models and underfitting with shallow ones. This project investigates the most effective architectural strategy for fine-grain classification tasks where data is limited.

## Why it Matters
Fine-grain recognition (e.g., distinguishing between 102 species of flowers) is a challenging task because categories share high visual similarity. Understanding when to use custom models versus pre-trained backbones is critical for practical computer vision applications.

## Constraints
* **Dataset Size**: Oxford Flowers-17 and Flowers-102 provide a limited number of samples per class.
* **Hardware Acceleration**: Optimized for **Apple Silicon** using the `mps` (Metal Performance Shaders) backend.
* **Architecture Complexity**: Finding a model size that captures fine features without overfitting.

## Approach
* **Scratch Training**: Developing and training custom CNN and ViT models from scratch on the target dataset.
* **Data Augmentation**: Implementation of random cropping, flipping, and normalization to improve generalization.
* **Transfer Learning**: Utilizing pre-trained SqueezeNet and GoogLeNet backbones with custom classification heads.

## Architecture
* **Custom CNN**: A 6-layer convolutional architecture with batch normalization and global average pooling.
* **Vision Transformer (ViT)**: A ViT-B/16 based transformer model adapted for $224 \times 224$ images.
* **SqueezeNet (Transfer Learning)**: Pre-trained SqueezeNet backbone with frozen feature weights and a custom linear head for 102 classes.

## Implementation Notes
* **Backend**: `torch.backends.mps` for local acceleration.
* **Data Loading**: Custom `torch.utils.data.Dataset` implementations for Oxford Flowers datasets.
* **Scheduling**: Use of `CosineAnnealingLR` to stabilize training for the transfer learning experiments.

## Results
| Approach | Model | Epochs | Test Accuracy |
| :--- | :--- | :---: | :---: |
| **Scratch** | Custom CNN | 50 | Underfit / Overfit |
| **Transfer Learning** | SqueezeNet | 10 | $54.10\%$ |

*Note: Models trained from scratch showed significant training-validation divergence despite heavy augmentation.*

## Trade-offs
* **Custom Models**: Highly interpretable and lightweight, but failed to reach competitive accuracy due to the complexity of the fine-grain task.
* **Transfer Learning**: Provides a superior feature extraction baseline but is limited by the fixed knowledge of the pre-trained weights.

## Reflection
Leveraging pre-trained models (even lightweight ones like SqueezeNet) is significantly more effective than training complex architectures like ViTs from scratch on small datasets.

## Tech Stack
* **Language**: Python
* **Libraries**: PyTorch, Torchvision, PIL, Matplotlib, NumPy
* **Acceleration**: Metal Performance Shaders (MPS)

## Project Type
AI / Computer Vision
