# RESNET  
This repository contains PyTorch code for training and testing two types of residual networks on the **CIFAR-10 dataset**. The project focuses on experimenting with **Direct Identity Residual Blocks** and **Resized Residual Blocks** to classify images into 10 categories with high accuracy.  
![Figure_1](https://github.com/user-attachments/assets/00423fb7-9fa2-45a6-99a1-070a67dbdedf)  

## Project Overview

In this project, we experiment with different residual blocks to classify images from the CIFAR-10 dataset. Residual blocks help in building deep neural networks by mitigating the vanishing gradient problem through shortcut connections. We test two residual architectures:

1. **Direct Identity Residual Block**: A basic residual block that uses identity shortcuts without resizing the input.
2. **Resized Residual Block**: A modified residual block that resizes the input channels and spatial dimensions before applying the shortcut.

Both networks are trained for 20 epochs using the CIFAR-10 dataset, which consists of 60,000 32x32 color images across 10 classes (e.g., airplanes, cars, birds, etc.).

## Architecture

### Direct Identity Residual Block
This architecture uses a straightforward residual block where the input dimensions remain unchanged. The block includes two 3x3 convolutional layers followed by batch normalization and ReLU activation. The original input is added back (skip connection) to the output of the block, which helps the model learn identity mappings easily.

### Resized Residual Block
This model incorporates a residual block where the input is downsampled both spatially and in terms of channels. The block consists of:
- A 3x3 convolution to downsample the input.
- A 1x1 convolution to adjust the number of channels.
- The shortcut connection is also resized to match the output shape of the block, ensuring proper addition.

## Installation

To run the code in this repository, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/bibasrairockz/RESNET.git
   cd resnet
  
