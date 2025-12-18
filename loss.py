import torch
import torch.nn.functional as F

def dice_loss_multiclass(y_pred, y_true, smooth=1e-6):
    """
    y_pred: logits, shape [B, C, H, W]
    y_true: integer mask, shape [B, H, W]
    """
    # convert logits → probs
    y_pred = torch.softmax(y_pred, dim=1)

    # one-hot encode mask → [B, C, H, W]
    C = y_pred.shape[1]
    y_true_1hot = F.one_hot(y_true, C).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    intersection = torch.sum(y_pred * y_true_1hot, dims)
    union = torch.sum(y_pred + y_true_1hot, dims)

    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()