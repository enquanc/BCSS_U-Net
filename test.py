import torch.nn as nn
import torch 
from tqdm import tqdm
from util import *
import argparse
from dataset import *
from model import *
from torch.utils.data import DataLoader
from model_copy import *


def test_model(model, test_dataloader, config):
    """
    修改版：使用 Global Accumulation 計算 mDice 和 mIoU
    解決 Batch 平均導致的分數虛高問題
    """
    device = config['device']
    model.eval()
    
    loss_fn = nn.CrossEntropyLoss()
    test_loss = 0.0
    num_batches = len(test_dataloader)
    
    # ### <--- 修改 1: 初始化全域計數器 (而不是分數累加器) ###
    # 我們需要知道總共有多少類別，先假設從 config 或 dataloader 取得，這裡動態偵測
    num_classes = config.get('out_channels', 22) # 預設 22
    
    # 紀錄每個類別的總統計量 (放在 GPU 上運算較快)
    total_inter = torch.zeros(num_classes).to(device)
    total_union = torch.zeros(num_classes).to(device) # For IoU
    total_pred_area = torch.zeros(num_classes).to(device) # For Dice
    total_target_area = torch.zeros(num_classes).to(device) # For Dice
    
    print(f"Starting Global Testing on {device}...")
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_dataloader, desc="Testing"):
            inputs = inputs.to(device)
            targets = targets.to(device) 

            # 1. 模型預測
            logits = model(inputs) 
            loss = loss_fn(logits, targets)
            test_loss += loss.item()
            
            # ### <--- 修改 2: 移除逐 Batch 的 metric 計算，改為計算 Intersection & Union ###
            preds = torch.argmax(logits, dim=1) # [B, H, W]
            
            # 展平以便計算
            preds = preds.view(-1)
            targets = targets.view(-1)
            
            # 排除 Ignore Index (通常是 255)
            valid_mask = (targets != 255)
            preds = preds[valid_mask]
            targets = targets[valid_mask]
            
            # 針對每個類別累加數值
            for c in range(num_classes):
                # 建立二元遮罩
                p_mask = (preds == c)
                t_mask = (targets == c)
                
                # 計算基礎統計量
                intersection = (p_mask & t_mask).sum()
                pred_area = p_mask.sum()
                target_area = t_mask.sum()
                union = pred_area + target_area - intersection
                
                # 累加到全域變數
                total_inter[c] += intersection
                total_union[c] += union
                total_pred_area[c] += pred_area
                total_target_area[c] += target_area

    # ### <--- 修改 3: 迴圈結束後，計算 Global Metrics ###
    avg_loss = test_loss / num_batches
    
    # 計算每個類別的 Dice 和 IoU
    # Dice = 2*I / (Pred + Target)
    # IoU = I / Union
    epsilon = 1e-6
    
    class_dice = (2.0 * total_inter + epsilon) / (total_pred_area + total_target_area + epsilon)
    class_iou = (total_inter + epsilon) / (total_union + epsilon)
    
    # 處理那些「完全沒出現過」的類別 (Union == 0)
    # 避免因為 epsilon 導致有一個非零的很小分數
    for c in range(num_classes):
        if total_union[c] == 0:
            class_iou[c] = float('nan') # 標記為無效
            class_dice[c] = float('nan')

    # 轉換為 List 以便計算平均
    dice_list = class_dice.cpu().tolist()
    iou_list = class_iou.cpu().tolist()
    
    # 定義 helper function 來算平均 (忽略 NaN)
    def nan_mean(values):
        valid_values = [v for v in values if not math.isnan(v)]
        return sum(valid_values) / len(valid_values) if valid_values else 0.0

    import math # 記得 import math

    # 計算含背景 (bg) 和不含背景 (nbg) 的平均
    # 假設 Class 0 是背景
    
    # nbg: 從 index 1 開始
    avg_mDice_nbg = nan_mean(dice_list[1:])
    avg_mIoU_nbg = nan_mean(iou_list[1:])
    
    # bg: 包含 index 0
    avg_mDice_bg = nan_mean(dice_list)
    avg_mIoU_bg = nan_mean(iou_list)

    # 準備詳細報告字典 (方便印出)
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
    
    # 選擇性印出每個類別的分數
    print("Per Class Dice Score:")
    for k, v in dice_metrics_detailed.items():
        print(f"{k}: {v:.4f}")
        
    return avg_loss, avg_mDice_nbg, avg_mDice_bg, avg_mIoU_nbg, avg_mIoU_bg


if __name__ =='__main__':
    

    # ==========================================
    # 執行方式
    # ==========================================



    # 1. 確保 config 裡有 device 設定

    parser = argparse.ArgumentParser(description='Train the model')

    # 1. Device (通常預設自動偵測，但也允許手動指定 'cuda:1' 等)
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
        # model = old_AttentionUNet(in_channels=test_config['in_channels'], out_channels=test_config['out_channels'], n_filters = test_config['n_filter'])
        model = AttentionUNet(in_channels=test_config['in_channels'], out_channels=test_config['out_channels'], n_filters = test_config['n_filter'])
        model.load_state_dict(torch.load(test_config['ckpt'], map_location=test_config['device']))
        model.to(test_config['device'])
        model.eval()

    # 2. 確保您有引入之前寫好的 metric function
    # from your_utils import dice_score_detailed, iou_score_multiclass

    # 3. 執行
    test_loss, test_dice_nbg, test_dice_bg, test_iou_nbg, test_iou_bg = test_model(model, test_dataloader, test_config)

    # python test.py   --model Unet --ckpt "checkpoints/Unet-weights.pth" --device cuda:1 --n_filter 32
    # python test.py   --model Attention-Unet --ckpt "checkpoints/Attention-Unet-weights.pth" --device cuda:1 --n_filter 32
    # python test.py   --model Unet --ckpt "checkpoints/512/Unet-weights.pth" --device cuda:1 --out_channels 22 --n_filter 32
    # python test.py   --model Attention-Unet --ckpt "checkpoints/512/Attention-Unet-weights.pth" --device cuda:1 --out_channels 22 --n_filter 32

