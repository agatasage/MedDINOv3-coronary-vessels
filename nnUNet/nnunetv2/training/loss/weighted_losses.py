import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WeightedDC_and_CE(nn.Module):
    def __init__(self, dc_loss, class_weights, weight_dice=1.0, weight_ce=1.0):
        super().__init__()
        self.dc_loss = dc_loss
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce

    def forward(self, preds, target):
        device = preds.device
        cw = self.class_weights.to(device)

        # --- one hot
        if target.ndim == 4 and target.shape[1] == 1:
            target = target.squeeze(1).long()
            target = F.one_hot(target, num_classes=preds.shape[1])
            target = target.permute(0, 3, 1, 2).float()

        # --- CE (weighted)
        ce = F.cross_entropy(preds, target.argmax(1) if target.ndim == 4 else target, weight=cw)

        # --- Dice (weighted?)
        softmax_preds = torch.softmax(preds, dim=1)
        target_onehot = target

        dims = (0, 2, 3)

        intersection = (softmax_preds * target_onehot).sum(dims)
        union = softmax_preds.sum(dims) + target_onehot.sum(dims)

        dice_per_class = (2 * intersection + 1e-5) / (union + 1e-5)

        weighted_dice = (cw * (1 - dice_per_class)).mean()

        return self.weight_ce * ce + self.weight_dice * weighted_dice
    
class BoundaryLoss(nn.Module):
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight

    def forward(self, preds, target):
        probs = torch.softmax(preds, dim=1)

        # --- one-hot
        if target.ndim == 4 and target.shape[1] == 1:
            target = target.squeeze(1).long()
            target = F.one_hot(target, num_classes=preds.shape[1])
            target = target.permute(0, 3, 1, 2).float()

        dx_pred = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :])
        dy_pred = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1])

        dx_gt = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        dy_gt = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])

        loss = F.l1_loss(dx_pred, dx_gt) + F.l1_loss(dy_pred, dy_gt)

        return self.weight * loss
    
class VesselLoss(nn.Module):
    def __init__(self, base_loss, class_weights, boundary_weight=0.1):
        super().__init__()
        self.base_loss = base_loss
        self.boundary = BoundaryLoss(boundary_weight)

    def forward(self, preds, target):
        loss = self.base_loss(preds, target)
        loss = loss + self.boundary(preds, target)
        return loss