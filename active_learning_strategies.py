#!/usr/bin/env python3
"""
SAM Active Learning Strategies Module
Contains implementations of uncertainty sampling, random sampling, mask difference, etc.
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
import logging
from tqdm import tqdm
from abc import ABC, abstractmethod

# Import ALiPy algorithms (optional)
try:
    import alipy
    from alipy.query_strategy import (
        QueryInstanceUncertainty,
        QueryInstanceRandom,
        QueryInstanceQUIRE,
    )
    ALIPY_AVAILABLE = True
except ImportError:
    ALIPY_AVAILABLE = False
    print("Warning: ALiPy not installed, using simplified active learning strategies")

class BaseActiveLearningStrategy(ABC):
    """Base class for active learning strategies"""
    
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def select_samples(self, model, dataset, unlabeled_indices, num_samples, device):
        """Select samples for annotation"""
        pass
    
    def calculate_uncertainty(self, logits):
        """Calculate uncertainty (based on entropy of prediction probabilities)"""
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        return entropy.mean().item()


class MaskDifferenceStrategy(BaseActiveLearningStrategy):
    
    def __init__(self):
        super().__init__("MaskDifference")
    
    def safe_squeeze_logits(self, logits):
        if logits.dim() == 4:  # [B, C, H, W]
            if logits.shape[0] == 1:  # batch size = 1
                logits = logits.squeeze(0)  # Remove batch dimension -> [C, H, W]
            if logits.shape[0] == 1:  # channel = 1
                logits = logits.squeeze(0)  # Remove channel dimension -> [H, W]
        elif logits.dim() == 3:  # [B, H, W] or [C, H, W]
            if logits.shape[0] == 1:  # First dimension = 1
                logits = logits.squeeze(0)  # Remove first dimension -> [H, W]
        return logits
    
    def calculate_dice_difference(self, prob1, prob2, threshold=0.6):
        mask1 = (prob1 > threshold).astype(np.float32)
        mask2 = (prob2 > threshold).astype(np.float32)
        
        intersection = np.sum(mask1 * mask2)
        dice = (2.0 * intersection) / (np.sum(mask1) + np.sum(mask2) + 1e-8)
        
        return 1.0 - dice
    
    def select_samples(self, model, dataset, unlabeled_indices, num_samples, device):
        # Vectorized GPU-accelerated scoring over mini-batches
        device_t = device if isinstance(device, torch.device) else torch.device(device)
        was_training = model.training
        model.eval()

        diffs = []
        idx_buffer = []
        batch_size = max(1, min(16, len(unlabeled_indices)))
        threshold = 0.6

        def dice_diff(m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
            # m1, m2: [B, C, H, W] binary masks
            intersect = (m1 * m2).sum(dim=(1, 2, 3))
            sum1 = m1.sum(dim=(1, 2, 3))
            sum2 = m2.sum(dim=(1, 2, 3))
            dice = (2.0 * intersect + 1e-8) / (sum1 + sum2 + 1e-8)
            return 1.0 - dice

        with torch.inference_mode():
            for start in range(0, len(unlabeled_indices), batch_size):
                batch_ids = unlabeled_indices[start:start + batch_size]
                images = []
                keep_mask = []
                for i_idx, idx in enumerate(batch_ids):
                    try:
                        sample = dataset[idx]
                        if 'image' not in sample:
                            keep_mask.append(False)
                            continue
                        img = sample['image']
                        if img.dim() == 3:
                            images.append(img)
                            keep_mask.append(True)
                        else:
                            keep_mask.append(False)
                    except Exception:
                        keep_mask.append(False)
                        continue

                if len(images) == 0:
                    continue

                images = torch.stack(images, dim=0).to(device_t, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=(device_t.type == 'cuda')):
                    # Prompt variants
                    out_no = model(images, True, 512, -1, 'point')
                    out_center = model(images, True, 512, 0, 'point')
                    out_rand = model(images, True, 512, 1, 'point')

                    # Extract logits and convert to probabilities
                    prob_no = torch.sigmoid(out_no['low_res_logits1']) if 'low_res_logits1' in out_no else None
                    prob_center = torch.sigmoid(out_center['low_res_logits2']) if 'low_res_logits2' in out_center else None
                    prob_rand = torch.sigmoid(out_rand['low_res_logits1']) if 'low_res_logits1' in out_rand else None

                if prob_no is None or prob_center is None or prob_rand is None:
                    # If any missing, skip this mini-batch
                    continue

                # Binary masks by threshold
                m_no = (prob_no > threshold).float()
                m_center = (prob_center > threshold).float()
                m_rand = (prob_rand > threshold).float()

                # Dice differences
                d_no_center = dice_diff(m_no, m_center)
                d_no_rand = dice_diff(m_no, m_rand)
                d_center_rand = dice_diff(m_center, m_rand)

                # Probability diffs
                p_no_center = torch.mean(torch.abs(prob_no - prob_center), dim=(1, 2, 3))
                p_no_rand = torch.mean(torch.abs(prob_no - prob_rand), dim=(1, 2, 3))
                p_center_rand = torch.mean(torch.abs(prob_center - prob_rand), dim=(1, 2, 3))

                # Weighted combination (same weights as original implementation)
                combined = (
                    0.25 * d_no_center +
                    0.20 * d_no_rand +
                    0.25 * d_center_rand +
                    0.10 * p_no_center +
                    0.10 * p_no_rand +
                    0.10 * p_center_rand
                )

                diffs.append(combined.detach().cpu())
                # Map back to original indices, exclude failed ones
                kept_ids = [j for j, ok in zip(batch_ids, keep_mask) if ok]
                idx_buffer.extend(kept_ids)

        if was_training:
            model.train()

        if len(diffs) == 0 or len(idx_buffer) == 0:
            return []

        scores = torch.cat(diffs, dim=0).numpy()
        # Select top-k
        topk = min(num_samples, scores.shape[0])
        sorted_idx = np.argsort(scores)[::-1][:topk]
        selected_indices = [idx_buffer[i] for i in sorted_idx]
        return selected_indices


class RandomStrategy(BaseActiveLearningStrategy):
    """Random sampling strategy"""
    
    def __init__(self):
        super().__init__("Random")
    
    def select_samples(self, model, dataset, unlabeled_indices, num_samples, device):
        return random.sample(unlabeled_indices, num_samples)


# Strategy factory
class ActiveLearningStrategyFactory:
    """Active learning strategy factory"""
    
    @staticmethod
    def create_strategy(strategy_name, **kwargs):
        """Create active learning strategy"""
        strategies = {
            'mask_difference': lambda: MaskDifferenceStrategy(),
        }
        
        if strategy_name in strategies:
            strategy_class = strategies[strategy_name]
            if callable(strategy_class):
                return strategy_class()
            else:
                return strategy_class()
        else:
            print(f"Unknown strategy {strategy_name}, using random strategy")
            return RandomStrategy()
    
    @staticmethod
    def get_available_strategies():
        """Get list of available strategies"""
        return ['random', 'mask_difference']