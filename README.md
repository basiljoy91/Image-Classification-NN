# Image Classification-NN

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.1-orange)
![Keras](https://img.shields.io/badge/Keras-2.6-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.1-green)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

A convolutional neural network (CNN) implementation for classifying images from the CIFAR-10 dataset using TensorFlow and Keras.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Examples](#examples)
- [Results](#results)
- [License](#license)

## Project Overview

This project implements a CNN to classify images from the CIFAR-10 dataset, which contains 60,000 32x32 color images across 10 classes. The model achieves good accuracy while being relatively lightweight.

**Key Components**:
- Image preprocessing and normalization
- CNN architecture with convolutional and pooling layers
- Model training and evaluation
- Image classification inference

**Technology Stack**:
- Python 3.7+
- TensorFlow 2.x
- Keras API
- OpenCV for image processing
- Matplotlib for visualization
- NumPy for numerical operations

## Features

### Core Features
- CIFAR-10 dataset loading and preprocessing
- CNN model implementation
- Training with validation
- Model saving and loading
- Single image classification

### Advanced Features
- Image visualization
- Accuracy and loss metrics
- GPU acceleration support
- Simple inference pipeline

## Installation

### Prerequisites
- Python 3.7 or later
- pip package manager
- (Optional) NVIDIA GPU with CUDA for accelerated training

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cifar10-classifier.git
cd cifar10-classifier
