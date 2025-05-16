# Astronomical Image Super-Resolution with Weighted Loss Functions

This repository contains the code for my undergraduate thesis project:  
**"Weighted Loss Function Design for Super-Resolution in Astronomical Imaging"**, based on and extending: Julia Netzel. Statistical stability of super-resolution for astronomical imaging. Master thesis, ETH Zürich, Department of Mathematics, September 2023.

## Project Overview

We build upon the SRCNN super-resolution framework and investigate the effectiveness of weighted loss functions—specifically, weighted MSE and weighted MAE—to enhance the restoration of central fine structures in astronomical images.

The project includes:
- 3 synthetic datasets with different PSF and noise levels: `4_001`, `10_001`, and `10_0015`
- Baseline and modified SRCNN implementations
- Custom loss functions: `WeightedMSELoss`, `WeightedMAELoss`
- Evaluation metrics: SSIM, Selective SSIM (SSSIM)

Also There is a paper attached in Chinese Version. 

# TO DO
Add some documentation and notes for easy use by others