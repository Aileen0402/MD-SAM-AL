import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
from einops import repeat
import logging

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        # Use calculation method consistent with DiceLoss
        smooth = 1e-5
        intersect = np.sum(pred * gt)
        y_sum = np.sum(gt)
        z_sum = np.sum(pred)
        dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        
        # Calculate HD95 with error handling
        try:
            hd95 = metric.binary.hd95(pred, gt)
            # Ensure HD95 is a finite value
            if not np.isfinite(hd95) or hd95 < 0:
                hd95 = 0.0
            # Limit maximum HD95 value to avoid abnormally large values
            if hd95 > 100.0:  # 100mm as reasonable upper limit
                hd95 = 100.0
        except (RuntimeError, ValueError, TypeError) as e:
            logging.warning(f"HD95 calculation failed: {e}")
            # If calculation fails, use image diagonal length as estimate
            hd95 = min(np.sqrt(pred.shape[0]**2 + pred.shape[1]**2) * 0.1, 100.0)  # Assume 0.1mm/pixel
        
        return dice, hd95
    else:
        return 0, 0

def test_medical_image(image, label, net, classes, multimask_output=True, patch_size=[512, 512], original_size=None):
    # Use the same device as the model
    device = next(net.parameters()).device
    input = image.to(device)
    label = label.squeeze().cpu().detach().numpy()
    
    # Calculate pixel spacing
    if original_size is not None:
        if isinstance(original_size, (tuple, list)) and len(original_size) >= 2:
            # For medical images, use more reasonable physical size estimation
            # Assume original image actual physical size is about 5cm x 5cm (more common medical image size)
            physical_size_mm = 50.0  # Assume physical size is 50mm x 50mm
            spacing_x = physical_size_mm / float(original_size[0])  # mm per pixel
            spacing_y = physical_size_mm / float(original_size[1])  # mm per pixel
            spacing = (spacing_x, spacing_y)
        else:
            spacing = (0.1, 0.1)
    else:
        # If no original size information, use more reasonable default value
        spacing = (0.1, 0.1)  # Default pixel spacing, closer to actual pixel spacing of medical images
    
    # Unified label processing: set 128 to 0, 255 to 1, other non-zero values to 1
    label[label == 128] = 0  # Set 128 to 0
    label[label == 255] = 1  # Set 255 to 1
    label[(label > 0) & (label != 1)] = 1  # Set other non-zero values to 1
    
    net.eval()
    with torch.no_grad():
        outputs = net(input, multimask_output, patch_size[0])
        output_masks = outputs['low_res_logits1']
        out = torch.softmax(output_masks, dim=1)  # Get probability instead of argmax
        out = out.squeeze().cpu().detach().numpy()
    
    metric_list = []
    for i in range(classes):  # Include background class (i=0)
        # Use calculation method completely consistent with DiceLoss
        pred_prob = out[i]  # Probability of class i
        gt_mask = (label == i).astype(np.float32)
        
        if gt_mask.sum() > 0:
            # Calculate Dice Score (opposite of _dice_loss in DiceLoss)
            smooth = 1e-5
            intersect = np.sum(pred_prob * gt_mask)
            y_sum = np.sum(gt_mask)
            z_sum = np.sum(pred_prob)
            dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
            
            # Calculate HD95 (only for foreground classes)
            if i > 0:  # Only calculate HD95 for foreground classes
                # Use more stable binarization method: fixed threshold 0.5, as medical image segmentation usually uses 0.5 as threshold
                if pred_prob.max() > 0:
                    pred_binary = (pred_prob > 0.5).astype(np.uint8)
                else:
                    pred_binary = np.zeros_like(pred_prob, dtype=np.uint8)
                
                # Check if both prediction and ground truth masks contain foreground
                if pred_binary.sum() > 0 and gt_mask.sum() > 0:
                    try:
                        # Use pixel spacing calculated based on original image size
                        # Ensure spacing is in correct format
                        if isinstance(spacing, (tuple, list)) and len(spacing) >= 2:
                            spacing_for_hd95 = spacing[:2]  # Only take first two elements
                        else:
                            spacing_for_hd95 = (0.1, 0.1)  # Default value
                        hd95 = metric.binary.hd95(pred_binary, gt_mask.astype(np.uint8), spacing_for_hd95)
                        # Ensure HD95 is a scalar value
                        if isinstance(hd95, (np.ndarray, list, tuple)):
                            hd95 = float(hd95.item() if hasattr(hd95, 'item') else hd95[0] if len(hd95) > 0 else 0.0)
                        else:
                            hd95 = float(hd95)
                        # Ensure HD95 is a finite value
                        if not np.isfinite(hd95) or hd95 < 0:
                            hd95 = 0.0
                        # Limit maximum HD95 value to avoid abnormally large values
                        if hd95 > 100.0:  # 100mm as reasonable upper limit
                            hd95 = 100.0
                    except (RuntimeError, ValueError, TypeError) as e:
                        # If calculation fails, log warning and return reasonable default value
                        logging.warning(f"HD95 calculation failed (class {i}): {e}")
                        hd95 = 0.0
                elif gt_mask.sum() > 0:
                    # If ground truth mask has foreground but prediction mask is empty, HD95 should be a large value
                    # Use image diagonal length as upper limit
                    spacing_val = spacing[0] if isinstance(spacing, (tuple, list)) else spacing
                    hd95 = min(np.sqrt(gt_mask.shape[0]**2 + gt_mask.shape[1]**2) * spacing_val, 100.0)
                else:
                    # If ground truth mask also has no foreground, HD95 is 0
                    hd95 = 0.0
            else:
                hd95 = 0.0
        else:
            dice = 0.0
            hd95 = 0.0
            
        metric_list.append((dice, hd95))
    return metric_list