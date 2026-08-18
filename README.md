# pytorch-fashionmnist-baseline



# PyTorch FashionMNIST Baseline Models

A beginner-friendly notebook that builds and trains simple neural networks on the **FashionMNIST** dataset using PyTorch.

This is the foundational step before moving to Convolutional Neural Networks (CNNs).

## What this notebook covers

- Loading the FashionMNIST dataset with `torchvision`
- Creating `DataLoader`s for training and testing
- Visualizing sample images and class labels
- Building two models:
  - **FashionMNISTModelV0** → Linear model (Flatten + Linear + Linear)
  - **FashionMNISTModelV1** → Non-linear model (Flatten + Linear + ReLU + Linear)
- Training loop with `CrossEntropyLoss` and `SGD`
- Custom `train_step` and `test_step` functions
- Model evaluation and accuracy comparison
- Training time measurement

## Models Overview

| Model                  | Architecture                          | Non-linearity | Approx. Test Accuracy |
|------------------------|---------------------------------------|---------------|-----------------------|
| FashionMNISTModelV0    | Flatten → Linear → Linear             | None          | ~83.4%                |
| FashionMNISTModelV1    | Flatten → Linear → ReLU → Linear      | ReLU          | ~84.8%                |



## How to run

Open the notebook in Google Colab, Jupyter, or VS Code.
Run all cells sequentially.
The first model trains on CPU by default (takes ~30–40 seconds for 3 epochs).

## Results (from the notebook)

Model V0 (linear): Test accuracy ≈ 83.4%
Model V1 (with ReLU): Test accuracy ≈ 84.8%
Training time (CPU, 3 epochs): ~31–38 seconds
