import os
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F
import imageio
from einops import repeat

class DiceLoss_Mask(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss_Mask, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, mask):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target * mask)
        y_sum = torch.sum(target * mask)
        z_sum = torch.sum(score * mask)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        
        # Create a mask to ignore the regions where target is 255
        mask = (target != 255).float()
        mask = mask.unsqueeze(1).repeat(1, self.n_classes, 1, 1)

        # Create a copy of target to avoid inplace modification
        target_copy = target.clone()
        target_copy[target_copy == 255] = 0
        target_copy = self._one_hot_encoder(target_copy)
        
        if weight is None:
            weight = [1] * self.n_classes
        
        assert inputs.size() == target_copy.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target_copy.size())
        
        class_wise_dice = []
        loss = 0.0
        
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target_copy[:, i], mask[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        
        return loss / self.n_classes