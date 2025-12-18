import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import os

def plot_sample(dataset, idx = 0):
    """
    Plot a sample image and its mask from the dataset.

    Args:
        dataset (Dataset): The custom BCSSDataset instance.
        idx (int): Index of the sample to be plotted.
    """
    image, mask = dataset[idx]
    image_np = image.permute(1, 2, 0).numpy()  # Convert from PyTorch tensor to numpy array
    mask_np = mask.squeeze().numpy()  # Remove channel dimension and convert to numpy array

    plt.figure(figsize=(12, 6))

    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title('Original Image')
    plt.axis('off')

    # Plot image with mask overlay
    plt.subplot(1, 2, 2)
    plt.imshow(image_np)
    plt.imshow(mask_np, alpha=0.6)  # Alpha controls the transparency
    plt.title('Image with Mask Overlay')
    plt.axis('off')

    plt.show()

####### mDice #######
def dice_score_multiclass(y_pred, y_true, smooth=1e-6, bg=False, ignore_index=255):
    """
    Robust Dice Score for High Cardinality Classes (e.g., 22 classes)
    """
    with torch.no_grad(): 
        y_pred_idx = torch.argmax(y_pred, dim=1)
        
        # 1. Process Ignore Index
        valid_mask = (y_true != ignore_index)
        y_pred_idx = y_pred_idx[valid_mask]
        y_true = y_true[valid_mask]
        
        C = y_pred.shape[1]
        results = {}
        foreground_scores = [] 

        for i in range(C):
            # p and t are 1D boolean tensor
            p_mask = (y_pred_idx == i)
            t_mask = (y_true == i)

            intersection = (p_mask & t_mask).sum().float()
            union = p_mask.sum() + t_mask.sum()


            if union == 0:
                score = 1.0 # pred and GT are 0，perfect !
            else:
                dice = (2 * intersection + smooth) / (union + smooth)
                score = dice.item() # call .item()

            results[f"Dice_Class_{i}"] = score

            # 4. mDice 
            if i > 0:
                foreground_scores.append(score)
            elif i == 0 and bg:
                foreground_scores.append(score)

        if len(foreground_scores) > 0:
            results["mDice"] = sum(foreground_scores) / len(foreground_scores)
        else:
            results["mDice"] = 0.0
    return results



####### mIou #######
def iou_multiclass(y_pred, y_true,smooth=1e-6,bg =False):

    y_pred_idx = torch.argmax(y_pred, dim=1)

    C = y_pred.shape[1]
    y_pred_1hot = F.one_hot(y_pred_idx, C).permute(0, 3, 1, 2).float()
    y_true_1hot = F.one_hot(y_true, C).permute(0, 3, 1, 2).float()

    class_ious = {}
    foreground_ious = []
    for i in range(C):
        # Flatten Global Batch IoU
        p = y_pred_1hot[:, i].contiguous().view(-1)
        t = y_true_1hot[:, i].contiguous().view(-1)

        intersection = (p * t).sum()
        # IoU 分母 = A + B - (A ∩ B)
        union = p.sum() + t.sum() - intersection

        if union == 0:
            iou = 1.0 # Empty target & Empty pred = Perfect
        else:
            iou_tensor = (intersection + smooth) / (union + smooth)
            iou = iou_tensor.item()

        class_ious[f"class_{i}_iou"] = iou

    # 4. Compute mIoU (Mean IoU)
    # Suppose Class 0 is background
    if not bg:
        foreground_ious = [v for k, v in class_ious.items() if "class_0" not in k]
    elif bg:
        foreground_ious = [v for k, v in class_ious.items()]

    if len(foreground_ious) > 0:
        m_iou = sum(foreground_ious) / len(foreground_ious)
    else:
        m_iou = 0.0
        
    class_ious["mIoU"] = m_iou
    
    return class_ious

def plot_learning_curves(train_loss, val_loss, title='--------', ylabel='Loss', save_path='./ckpt'):
    plt.style.use('ggplot')
    plt.rcParams['text.color'] = '#333333'

    fig, axis = plt.subplots(1, 1, figsize=(10, 6))

    # Plot training and validation loss (NaN is used to offset epochs by 1)
    if train_loss is not None and len(train_loss) > 0:
        axis.plot([np.NaN] + train_loss, color='#636EFA', 
                  marker='o', linestyle='-', linewidth=2, 
                  markersize=5, label='Training Loss')
        
    axis.plot([np.NaN] + val_loss,   color='#EFA363', 
              marker='s', linestyle='-', linewidth=2, 
              markersize=5, label=f'Validation {ylabel}')

    # Adding title, labels and formatting
    axis.set_title(title, fontsize=16)
    axis.set_xlabel('Epoch', fontsize=14)
    axis.set_ylabel(ylabel, fontsize=14, rotation=0, labelpad=20)

    # axis.set_ylim(0, 10)
    
    axis.legend(fontsize=12)
    axis.grid(True, which='both', linestyle='--', linewidth=0.5)

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    plt.show()
    plt.savefig(f'{save_path}/{title}.png')

def visualize_prediction(model, dataloader, device, num_samples=3):
    model.eval()
    inputs, targets = next(iter(dataloader))
    inputs = inputs.to(device)
    
    with torch.no_grad():
        logits = model(inputs)
        preds = torch.argmax(logits, dim=1) # [B, H, W]
        
    # Turn to CPU
    inputs = inputs.cpu()
    targets = targets.cpu()
    preds = preds.cpu()
    
    # Plot
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    for i in range(num_samples):
        # Original figure
        # Suppose inputs is [C, H, W]
        img = inputs[i].permute(1, 2, 0).numpy()
        # Process range
        img = (img - img.min()) / (img.max() - img.min())
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(targets[i], cmap='jet', vmin=0, vmax=2)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(preds[i], cmap='jet', vmin=0, vmax=2)
        axes[i, 2].set_title("Model Prediction")
        axes[i, 2].axis('off')
        
    plt.tight_layout()
    plt.show()

def display_test_sample(model, test_input, test_target, device):
    model.eval()
    test_input, test_target = test_input.to(device), test_target.to(device)

    # ---------------------------------------------------
    # 1. Get model prediction：Multi-class segmentation → softmax + argmax
    # ---------------------------------------------------
    with torch.no_grad():
        logits = model(test_input)
        probs = torch.softmax(logits, dim=1)                # [B, C, H, W]
        pred_mask = torch.argmax(probs, dim=1)              # [B, H, W]

    # ---------------------------------------------------
    # 2. Convert to numpy
    # ---------------------------------------------------
    image       = test_input[0].detach().cpu().permute(1,2,0).numpy()
    gt_mask     = test_target[0].detach().cpu().numpy()
    pred_mask   = pred_mask[0].detach().cpu().numpy()

    # ---------------------------------------------------
    # 3. plotting config
    # ---------------------------------------------------
    plt.rcParams['figure.facecolor'] = '#171717'
    plt.rcParams['text.color']       = '#DDDDDD'

    # ---------------------------------------------------
    # 4. Plot
    # ---------------------------------------------------
    plt.figure(figsize=(15,5))

    # image
    plt.subplot(1,3,1)
    plt.title("H&E Image")
    plt.imshow(image.astype(np.uint8))
    plt.axis("off")

    # GT mask
    plt.subplot(1,3,2)
    plt.title("Ground Truth Mask")
    plt.imshow(gt_mask, cmap="tab20")
    plt.axis("off")

    # pred mask
    plt.subplot(1,3,3)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask, cmap="tab20")
    plt.axis("off")

    plt.show()

    # ---------------------------------------------------
    # 5. Overlay Version
    # ---------------------------------------------------
    plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    plt.title("Overlay: Ground Truth")
    plt.imshow(image.astype(np.uint8))
    plt.imshow(gt_mask, cmap="tab20", alpha=0.35)
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Overlay: Prediction")
    plt.imshow(image.astype(np.uint8))
    plt.imshow(pred_mask, cmap="tab20", alpha=0.35)
    plt.axis("off")

    plt.show()
import matplotlib.pyplot as plt


def show_class_mapping(dataset, index=0):
    image, mask = dataset[index]
    
    # Change format
    if hasattr(image, 'permute'):
        image = image.permute(1, 2, 0).numpy() # (C, H, W) -> (H, W, C)
    if hasattr(mask, 'numpy'):
        mask = mask.numpy()

    # show for three class
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original figure 
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    
    # Class 0 ( Usually background )
    ax[1].imshow(mask == 0, cmap='gray')
    ax[1].set_title("Class 0 (Background?)")
    
    # Class 1 
    ax[2].imshow(mask == 1, cmap='gray')
    ax[2].set_title("Class 1 (Tumor?)")
    
    # Class 2 
    ax[3].imshow(mask == 2, cmap='gray')
    ax[3].set_title("Class 2 (Stroma?)")
    
    plt.show()


### Use
# show_class_mapping(train_dataset, index=433)