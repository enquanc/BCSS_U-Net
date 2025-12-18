import torch.nn as nn
import torch 
from tqdm import tqdm
from util import *
import argparse
from dataset import *
from model import *
from torch.utils.data import DataLoader
import math

def test_model(model, test_dataloader, config):
    device = config['device']
    model.eval()
    
    loss_fn = nn.CrossEntropyLoss()
    test_loss = 0.0
    num_batches = len(test_dataloader)
    
    num_classes = config.get('out_channels', 22) # Default 22 class
    
    # Record each class total statistics  
    total_inter = torch.zeros(num_classes).to(device)
    total_union = torch.zeros(num_classes).to(device) # For IoU
    total_pred_area = torch.zeros(num_classes).to(device) # For Dice
    total_target_area = torch.zeros(num_classes).to(device) # For Dice
    
    print(f"Starting Global Testing on {device}...")
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_dataloader, desc="Testing"):
            inputs = inputs.to(device)
            targets = targets.to(device) 

            # 1. Model predict
            logits = model(inputs) 
            loss = loss_fn(logits, targets)
            test_loss += loss.item()
            
            # Compute Intersection & Union
            preds = torch.argmax(logits, dim=1) # [B, H, W]
            
            # flatten to compute
            preds = preds.view(-1)
            targets = targets.view(-1)
            
            # Ignore Index (Usually 255)
            valid_mask = (targets != 255)
            preds = preds[valid_mask]
            targets = targets[valid_mask]
            
            # Accumulate each class 
            for c in range(num_classes):
                # Build binary mask
                p_mask = (preds == c)
                t_mask = (targets == c)
                
                # Compute statistics
                intersection = (p_mask & t_mask).sum()
                pred_area = p_mask.sum()
                target_area = t_mask.sum()
                union = pred_area + target_area - intersection
                
                # Accumulate to global variable
                total_inter[c] += intersection
                total_union[c] += union
                total_pred_area[c] += pred_area
                total_target_area[c] += target_area

    # Compute global metrics
    avg_loss = test_loss / num_batches
    
    # Compute each class's Dice and IoU
    # Dice = 2*I / (Pred + Target)
    # IoU = I / Union
    epsilon = 1e-6
    
    class_dice = (2.0 * total_inter + epsilon) / (total_pred_area + total_target_area + epsilon)
    class_iou = (total_inter + epsilon) / (total_union + epsilon)
    
    # Process "Never show " class ( Union == 0 )
    # To avoid the very small nmuber because the epsilon
    for c in range(num_classes):
        if total_union[c] == 0:
            class_iou[c] = float('nan') # Note nan to ignore
            class_dice[c] = float('nan')

    # Turn into list for average 
    dice_list = class_dice.cpu().tolist()
    iou_list = class_iou.cpu().tolist()
    
    # Define helper function to compute average (Ignore nan)
    def nan_mean(values):
        valid_values = [v for v in values if not math.isnan(v)]
        return sum(valid_values) / len(valid_values) if valid_values else 0.0



    # Compute the average ( with background and without background )
    # Suppose Class 0 is background
    
    # no background : start from 1 index
    avg_mDice_nbg = nan_mean(dice_list[1:])
    avg_mIoU_nbg = nan_mean(iou_list[1:])
    
    # background : Include index 0
    avg_mDice_bg = nan_mean(dice_list)
    avg_mIoU_bg = nan_mean(iou_list)

    # Save detail dict result 
    dice_metrics_detailed = {f"Class_{i}": d for i, d in enumerate(dice_list)}

    print("\n" + "="*50)
    print("       GLOBAL TEST RESULTS       ")
    print("="*50)
    print(f"Avg Loss  (CE)                : {avg_loss:.4f}")
    print(f"Global mDice (No Background)  : {avg_mDice_nbg:.4f}")
    print(f"Global mDice (With Background): {avg_mDice_bg:.4f}")
    print(f"Global mIoU  (No Background)  : {avg_mIoU_nbg:.4f}")
    print(f"Global mIoU  (With Background): {avg_mIoU_bg:.4f}")
    print("="*50)
    
    # Print each class score
    print("Per Class Dice Score:")
    for k, v in dice_metrics_detailed.items():
        print(f"{k}: {v:.4f}")
        
    return avg_loss, avg_mDice_nbg, avg_mDice_bg, avg_mIoU_nbg, avg_mIoU_bg


if __name__ =='__main__':

    parser = argparse.ArgumentParser(description='Train the model')

    # 1. Device
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', type=str, default=default_device, help='Device to use (cuda/cpu)')

    # 2. Perparameters
    parser.add_argument('--model', type=str, default='Unet', help='Select which model')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--ckpt', type=str, default="attention-unet-weights.pth", help='Directory to use model ckpt')
    parser.add_argument('--n_filter', type=int, default=32, help='filter size')
    parser.add_argument('--in_channels', type=int, default=3, help='number of input channels')
    parser.add_argument('--out_channels', type=int, default=3, help='number of output channels')

    
    args = parser.parse_args()
    test_config = vars(args)

    if test_config['out_channels'] == 3 :
        _, _, test_dataset = create_dataset(train_image_path = 'dataset/archive/BCSS/train/', val_image_path = 'dataset/archive/BCSS/val/', test_image_path = 'dataset/archive/BCSS/test/',
                        train_mask_path = 'dataset/archive/BCSS/train_mask/', val_mask_path = 'dataset/archive/BCSS/val_mask/') 
    elif test_config['out_channels'] == 22 :
        _, _, test_dataset = create_dataset(train_image_path = 'dataset/archive/BCSS_512/train_512/', val_image_path = 'dataset/archive/BCSS_512/val_512/', test_image_path = 'dataset/archive/BCSS/test/',
                        train_mask_path = 'dataset/archive/BCSS_512/train_mask_512/', val_mask_path = 'dataset/archive/BCSS_512/val_mask_512/') 

    test_dataloader = DataLoader(test_dataset, batch_size=test_config['batch_size'], shuffle=False, num_workers=8, pin_memory=True)


    if test_config['model'] =='Unet':
        # Create UNet model and count params
        model = UNet(in_channels=test_config['in_channels'], out_channels=test_config['out_channels'], n_filters = test_config['n_filter'])
        model.load_state_dict(torch.load(test_config['ckpt'], map_location=test_config['device']))
        model.to(test_config['device'])
        model.eval()
    elif test_config['model'] == 'Attention-Unet':
        model = AttentionUNet(in_channels=test_config['in_channels'], out_channels=test_config['out_channels'], n_filters = test_config['n_filter'])
        model.load_state_dict(torch.load(test_config['ckpt'], map_location=test_config['device']))
        model.to(test_config['device'])
        model.eval()

    # 3. Run
    test_loss, test_dice_nbg, test_dice_bg, test_iou_nbg, test_iou_bg = test_model(model, test_dataloader, test_config)


