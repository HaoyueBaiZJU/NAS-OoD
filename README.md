# NAS-OoD: Neural Architecture Search for Out-of-Distribution Generalization

## Outline

Recent advances on Out-of-Distribution (OoD) generalization reveal the robustness of deep learning models against different kinds of distribution shifts in real-world applications.
In this work, we propose robust Neural Architecture Search for OoD generalization (NAS-OoD), which optimizes the architecture with respect to its performance on the generated OoD data by gradient descent.
Extensive experimental results show that NAS-OoD achieves superior performance on various OoD generalization benchmarks with deep models having a much fewer number of parameters.

### Prerequisites

Python3.6. and the following packages are required to run the scripts:

- Python >= 3.7

- PyTorch >= 1.1 and torchvision

- CVXPY

- Mosek

### Code Structure

 - train_search_single.py
 - nas_ood_single
 - dataloader


### Demonstrations on NICO

bash main_search_onestage.sh

