import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_edge_weight_map(image_tensor, method='sobel'):
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    sobel_x = sobel_x.to(image_tensor.device)
    sobel_y = sobel_y.to(image_tensor.device)

    grad_x = F.conv2d(image_tensor, sobel_x, padding=1)
    grad_y = F.conv2d(image_tensor, sobel_y, padding=1)
    
    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    weight_map = grad_mag / (grad_mag.max() + 1e-6)
    # weight_map = 1.0 + 4.0 * weight_map  # 细节区域权重范围 [1, 5] wl_01
    # weight_map = 1.0 + 2.0 * weight_map  # 细节区域权重范围 [1, 3] wl_02
    weight_map = 1.0 + 0.5 * weight_map  # 细节区域权重范围 [1, 3] wl_03
    return weight_map

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, pred, target):
        with torch.no_grad():
            weight_map = compute_edge_weight_map(target)
        loss = weight_map * (pred - target) ** 2
        return loss.mean()
