# ✅ Center-Weighted MSELoss for Astronomy Images
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_center_weight_map(image_tensor, alpha=0.5):
    """
    Emphasize the center region of the image.
    Input: image_tensor shape (B, C, H, W) or (C, H, W)
    Output: weight_map shape (B, 1, H, W)
    """
    if image_tensor.ndim == 3:  # e.g. (C, H, W)
        image_tensor = image_tensor.unsqueeze(0)  # → (1, C, H, W)
    elif image_tensor.ndim == 2:  # just in case
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # → (1, 1, H, W)
        
    B, C, H, W = image_tensor.shape
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    yy = yy.to(image_tensor.device)
    xx = xx.to(image_tensor.device)

    center_y, center_x = H / 2, W / 2
    distance = torch.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    norm_dist = distance / distance.max()  # [0, 1]
    weight = 1.0 + alpha * (1.0 - norm_dist)  # Center: 1+α, Edge: 1.0

    return weight.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)

class CenterWeightedMSELoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        with torch.no_grad():
            weight_map = compute_center_weight_map(target, alpha=self.alpha)
        loss = weight_map * (pred - target) ** 2
        return loss.mean()
