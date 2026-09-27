# CIFAR-10 ResNet Classification

This project trains a ResNet-based image classifier on the CIFAR-10 dataset and explores several optimization and regularization techniques to improve generalization and training stability.

The main focus is on combining data augmentation, learning-rate scheduling, and regularization techniques within a standard image classification pipeline.

## Project Overview

The goal of this project is to classify images from the CIFAR-10 dataset using a ResNet architecture.

The project investigates how several training strategies affect model performance:

- Learning-rate warmup
- Label smoothing
- TrivialAugment
- Cosine annealing
- CIFAR-10 normalization

## Dataset

CIFAR-10 contains:

- 60,000 color images
- Image size: 32 × 32
- 10 classes

The standard CIFAR-10 train/test split was used.

Input images were normalized using the CIFAR-10 mean and standard deviation.

## Model Architecture

A ResNet model was used as the image classifier.

Residual connections help deep networks learn more effectively by allowing information and gradients to flow through shortcut connections.

The model was trained from scratch on CIFAR-10.

## Optimization Strategies

### Learning-Rate Warmup

A 5-epoch learning-rate warmup was applied at the beginning of training.

Instead of immediately training with the full learning rate, the learning rate gradually increased during the first few epochs.

This helped stabilize the early training process.

### Label Smoothing

Label smoothing was used to reduce model overconfidence.

The classification targets were softened so that the model would generalize better to unseen images instead of becoming overly confident on the training set.

### TrivialAugment

TrivialAugment was applied to the training images.

Random transformations such as rotations and crops were used to increase training-data diversity and reduce overfitting.

### Cosine Annealing

After the warmup stage, cosine annealing was used to gradually reduce the learning rate as training progressed.

This allowed the model to make larger updates early in training and smaller refinements later.

## Experimental Setup

The reported training configuration included:

- Epochs: 200
- Initial Learning Rate: 0.1
- Warmup Epochs: 5
- Batch Size: 128
- Learning-Rate Schedule: Cosine Annealing

## Training Pipeline

The overall workflow was:

1. Load CIFAR-10
2. Apply normalization
3. Apply training-time augmentation
4. Initialize the ResNet model
5. Apply learning-rate warmup
6. Train with label smoothing
7. Reduce the learning rate using cosine annealing
8. Track training and validation performance

## Technologies

- Python
- PyTorch
- Torchvision
- ResNet
- CIFAR-10
- TrivialAugment
- NumPy
- Matplotlib

## Repository Structure

```text
cifar10-resnet-classification/
├── README.md
├── notebooks/
│   └── cifar10_resnet.ipynb
├── src/
│   └── train.py
├── results/
│   ├── training_curve.png
│   └── validation_accuracy.png
├── docs/
│   └── report.pdf
└── .gitignore
