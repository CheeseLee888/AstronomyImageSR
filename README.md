# Astronomical Image Super-Resolution with Weighted Loss Functions

This repository contains the code for my undergraduate thesis project:  
"Weighted Loss Function Design for Super-Resolution in Astronomical Imaging".  
The project is based on and extends Julia Netzel, ETH Zürich, Department of Mathematics, 2023.  

## Project Overview

We build upon the SRCNN super-resolution framework and investigate the effectiveness of weighted loss functions, specifically weighted MSE and weighted MAE, to enhance the restoration of central fine structures in astronomical images. We also compare with the Richardson–Lucy deconvolution algorithm.  

The project includes:  
- Synthetic datasets with different PSF and noise levels  
- Baseline and modified SRCNN implementations  
- Custom loss functions (Weighted MSE, Weighted MAE)  
- Evaluation metrics: PSNR, SSIM, Selective SSIM (SSSIM)  

A paper in Chinese version is attached.  

## Implementation Highlights

- Implemented SRCNN in PyTorch for astronomical image super-resolution  
- Designed and tested center-weighted MSE and MAE loss functions  
- Benchmarked against the Richardson–Lucy classical deconvolution algorithm  
- Evaluated performance using PSNR, SSIM, and SSSIM  
- Demonstrated improvements in detail consistency when using weighted losses  

## Quick Demo

The repository provides a small demo dataset (ten FITS images). Results can be reproduced with the following steps:

```bash
git clone https://github.com/yourname/AstronomyImageSR.git
cd AstronomyImageSR
pip install -r requirements.txt
bash run_demo.sh
```

## Representative Thesis Results

To complement the demo, we include selected results from the full thesis experiments on COSMOS NICMOS HST datasets.  

### Example Images (Dataset 4.001)

| HR Image | LR Image | RL Reconstruction | SRCNN (MSE) Reconstruction |
|----------|----------|-------------------|----------------------------|
| ![](assets/hr.png) | ![](assets/lr_4_001.png) | ![](assets/rl_4_001.png) | ![](assets/srcnn_mse_4_001.png) |

<!-- ### Distribution of SSIM and SSSIM Scores

Representative score distributions for different loss functions (dataset 4.001). Weighted loss functions improve the SSSIM metric, which emphasizes fine structural similarity.

| MSE (no weight) | MAE (no weight) | MSE weighted α=0.2 | MAE weighted α=0.2 |
|-----------------|-----------------|---------------------|---------------------|
| ![](assets/mse_no_weight.png) | ![](assets/mae_no_weight.png) | ![](assets/mse_weighted.png) | ![](assets/mae_weighted.png) | -->

### Quantitative Metrics (Dataset 4.001)

| Loss Function    | SSIM Mean | SSIM SD | SSSIM Mean | SSSIM SD |
|------------------|-----------|---------|------------|----------|
| MSE              | 0.983     | 0.027   | 0.791      | 0.230    |
| Weighted MSE α=2 | 0.985     | 0.027   | 0.792      | 0.230    |
| Weighted MSE α=4 | 0.986     | 0.027   | 0.797      | 0.224    |
| MAE              | 0.989     | 0.032   | 0.766      | 0.273    |
| Weighted MAE α=2 | 0.989     | 0.031   | 0.771      | 0.243    |
| Weighted MAE α=4 | 0.989     | 0.032   | 0.778      | 0.239    |

Weighted MAE consistently improves SSSIM, showing better preservation of fine structures compared to non-weighted losses.
