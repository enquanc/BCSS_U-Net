import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from loss import *
from tqdm import tqdm
from util import *
import pandas as pd
from dataset import *
from model import *
from torch.utils.data import DataLoader
import argparse
from torch.cuda.amp import autocast, GradScaler

def train_model(model, train_dataloader, val_dataloader, config, verbose=True, loss_fn = nn.CrossEntropyLoss()):
    device = config['device']
    n_epochs = config['n_epochs']
    learning_rate = config['learning_rate']
    lr_decay_factor = config['lr_decay_factor']
    save_dir = config['save_dir']   # <--- save path
    model_name = config['model']    # <--- model name

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay = 1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_decay_factor, patience=3, verbose=True)
    scaler = GradScaler()

    loss_ce = loss_fn
    loss_dice = dice_loss_multiclass

    history = {
            'train_ce_loss': [], 'train_dice_loss': [],'train_total_loss': [],
            'val_ce_loss': [], 'val_dice_loss': [],'val_total_loss': [],  'val_mDice': [] # mDice
        }
    
    best_val_loss = float('inf')  

    print("Starting Training...")
    for epoch in tqdm(range(1, n_epochs + 1)):
        # ----------------------
        # Training Phase
        # ----------------------
        model.train()
        train_running_loss = 0
        train_running_dice_loss = 0
        train_loop = tqdm(train_dataloader, desc=f"Epoch {epoch}/{n_epochs} [Train]", leave=False)

        for train_batch_idx, (train_inputs, train_targets) in enumerate(train_loop):

            train_inputs, train_targets = train_inputs.to(device), train_targets.to(device)
            
            optimizer.zero_grad()

            with autocast():
                train_preds = model(train_inputs)

                train_ce_loss = loss_ce(train_preds, train_targets)
                train__dice_loss = loss_dice(train_preds,  train_targets)
                total_loss = train_ce_loss + train__dice_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()


            # total_loss.backward()
            # optimizer.step()

            train_running_loss += train_ce_loss.item()
            train_running_dice_loss += train__dice_loss.item()

            train_loop.set_postfix(loss=total_loss.item())

        # Compute Epoch average Loss
        avg_train_ce_loss = train_running_loss / len(train_dataloader)
        avg_train_dice_loss = train_running_dice_loss / len(train_dataloader)
        avg_train_total_loss = avg_train_ce_loss + avg_train_dice_loss

        history['train_ce_loss'].append(avg_train_ce_loss)
        history['train_dice_loss'].append(avg_train_dice_loss)
        history['train_total_loss'].append(avg_train_total_loss)

        # ----------------------
        # Validation Phase
        # ----------------------
        model.eval()
        val_running_loss = 0
        val_running_dice_loss = 0
        val_running_mDice = 0
    
        val_loop = tqdm(val_dataloader, desc=f"Epoch {epoch}/{n_epochs} [Val]", leave=False)

        with torch.no_grad():
            for idx, (val_inputs, val_targets) in enumerate(val_loop):

                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                
                val_preds = model(val_inputs)
                
                val_ce_loss = loss_ce(val_preds, val_targets)
                val_dice_loss = loss_dice(val_preds, val_targets)
                val_running_loss += val_ce_loss.item()
                val_running_dice_loss += val_dice_loss.item()
                
                metrics = dice_score_multiclass(val_preds, val_targets) 
                val_running_mDice += metrics['mDice']

        avg_val_ce_loss = val_running_loss / len(val_dataloader)
        avg_val_dice_loss = val_running_dice_loss / len(val_dataloader)
        avg_val_mDice = val_running_mDice / len(val_dataloader) #  Compute mDice
        avg_val_total_loss = avg_val_ce_loss + avg_val_dice_loss

        history['val_ce_loss'].append(avg_val_ce_loss)
        history['val_dice_loss'].append(avg_val_dice_loss)
        history['val_total_loss'].append(avg_val_total_loss)
        history['val_mDice'].append(avg_val_mDice)


        # Save Best Model

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        # Judge best model or not
        if avg_val_total_loss < best_val_loss:
            print(f"Epoch {epoch}: Validation Loss improved from {best_val_loss:.4f} to {avg_val_total_loss:.4f}. Saving best model...")
            best_val_loss = avg_val_total_loss
            
            # Save best model, Usually name with '_best'
            best_model_path = os.path.join(save_dir, f"{model_name}_best.pth")
            
            # save model
            try:
                save_model(model, best_model_path) 
            except NameError:
                torch.save(model.state_dict(), best_model_path)

        # ----------------------
        # Scheduler Step & Logging
        # ----------------------
        current_lr = optimizer.param_groups[0]['lr']
        # Let Scheduler choose LR based on  Val Loss
        scheduler.step(avg_val_total_loss)

        if verbose:
            print(f"Epoch {epoch} | LR: {current_lr:.6f}")
            print(f"  Train Total Loss: {avg_train_total_loss:.4f} (CE: {avg_train_ce_loss:.4f} + DiceLoss: {avg_train_dice_loss:.4f})")
            print(f"  Val Total Loss: {avg_val_total_loss:.4f} (CE: {avg_val_ce_loss:.4f} + DiceLoss: {avg_val_dice_loss:.4f})")
            print("-" * 60)

    print("Training complete.")
    return history



if __name__ =='__main__':


    parser = argparse.ArgumentParser(description='Train the model')

    # 1. Device
    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', type=str, default=default_device, help='Device to use (cuda/cpu)')

    # 2. Training Hyperparameters
    parser.add_argument('--n_epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-3, help='Initial learning rate')
    parser.add_argument('--lr_decay_factor', type=float, default=0.85, help='Factor for learning rate decay')
    parser.add_argument('--model', type=str, default='Unet', help='Select which model')

    # 3. Other setting
    parser.add_argument('--save_dir', type=str, default='./checkpoints/', help='Directory to save models')
    # parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--output_channels', type=int, default=3, help='Number of output channels')
    parser.add_argument('--n_filters', type=int, default=32, help='Number of filters')
    
    args = parser.parse_args()
    train_config = vars(args)

    if train_config['output_channels'] == 3 :
        train_dataset, val_dataset, test_dataset = create_dataset(train_image_path = 'dataset/archive/BCSS/train/', val_image_path = 'dataset/archive/BCSS/val/', test_image_path = 'dataset/archive/BCSS/test/',
                        train_mask_path = 'dataset/archive/BCSS/train_mask/', val_mask_path = 'dataset/archive/BCSS/val_mask/') 
    elif train_config['output_channels'] == 22 :
                train_dataset, val_dataset, test_dataset = create_dataset(train_image_path = 'dataset/archive/BCSS_512/train_512/', val_image_path = 'dataset/archive/BCSS_512/val_512/', test_image_path = 'dataset/archive/BCSS/test/',
                        train_mask_path = 'dataset/archive/BCSS_512/train_mask_512/', val_mask_path = 'dataset/archive/BCSS_512/val_mask_512/') 


    if train_config['model'] =='Unet':
        # Create UNet model and count params
        model = UNet(in_channels=3, out_channels=train_config['output_channels'], n_filters=train_config['n_filters'])
    elif train_config['model'] == 'Attention-Unet':
        model = AttentionUNet(in_channels=3, out_channels=train_config['output_channels'], n_filters=train_config['n_filters'])

    print(count_parameters(model))

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    # Train model
    training_history = train_model(model, train_dataloader, val_dataloader, train_config, verbose=True)

    if not os.path.exists(train_config["save_dir"]):
        os.makedirs(train_config["save_dir"], exist_ok=True)
    # Save weights
    save_model(model, f'{train_config["save_dir"]}{train_config["model"]}-weights_last.pth')

    plot_learning_curves(training_history['train_ce_loss'], 
                     training_history['val_ce_loss'], 
                     title = f'{train_config["model"]} Cross Entropy Loss',
                     save_path=train_config["save_dir"],
                     ylabel='Loss'
                     )


    plot_learning_curves(training_history['train_dice_loss'], 
                        training_history['val_dice_loss'], 
                        title = f'{train_config["model"]} Dice Loss',
                        save_path=train_config["save_dir"],
                        ylabel='Loss')


    plot_learning_curves(training_history['train_total_loss'], 
                        training_history['val_total_loss'], 
                        title = f'{train_config["model"]} Total Loss (CE + Dice)',
                        save_path=train_config["save_dir"],
                        ylabel='Loss'
                        )


    plot_learning_curves(None, 
                        training_history['val_mDice'], 
                        save_path=train_config["save_dir"],
                        title = f'{train_config["model"]} mDice Score(Validation Only)',
                        ylabel='Dice Score'
                        )

    # Command
    # docker exec -it enquanc_DL /bin/bash
    # python train.py  --n_epochs 30 
    # --batch_size 64 --learning_rate 5e-3 --lr_decay_factor 0.9 
    # --model Unet --device cuda:5 --save_dir ./checkpoints/1214_
    # --n_filters 64

    # python train.py  --n_epochs 20
    #  --batch_size 32 --learning_rate 5e-3 --lr_decay_factor 0.9 
    # --model Attention-Unet --device cuda:4 --save_dir ./checkpoints/1214_
    # --n_filters 64

    # python train.py  --n_epochs 50 --batch_size 32 --learning_rate 1e-3 --lr_decay_factor 0.7 --model Unet --save_dir ./checkpoints/wd-2/1217_64n --n_filters 64 --device cuda:x
    # python train.py  --n_epochs 50 --batch_size 32 --learning_rate 1e-3 --lr_decay_factor 0.7 --model Unet --save_dir ./checkpoints/wd-2/1217_32n --n_filters 32 --device cuda:x
    # python train.py  --n_epochs 50 --batch_size 32 --learning_rate 1e-3 --lr_decay_factor 0.7 --model Attention-Unet --save_dir ./checkpoints/wd-2/1217_64n --n_filters 64 --device cuda:x 
    # python train.py  --n_epochs 50 --batch_size 32 --learning_rate 1e-3 --lr_decay_factor 0.7 --model Attention-Unet --save_dir ./checkpoints/wd-2/1217_32n --n_filters 32 --device cuda:x 
    # python train.py  --n_epochs 50 --batch_size 16 --learning_rate 1e-3 --lr_decay_factor 0.7 --model Unet --save_dir ./checkpoints/512/wd-2/1217_64n --n_filters 64 --output_channels 22 --device cuda:x
    # python train.py  --n_epochs 50 --batch_size 32 --learning_rate 1e-3 --lr_decay_factor 0.7 --model Unet --save_dir ./checkpoints/512/wd-2/1217_32n --n_filters 32 --output_channels 22 --device cuda:x
    # python train.py  --n_epochs 50 --batch_size 4 --learning_rate 1e-3 --lr_decay_factor 0.7 --model Attention-Unet --save_dir ./checkpoints/512/wd-2/1217_64n --n_filters 64 --output_channels 22 --device cuda:x 
    # python train.py  --n_epochs 50 --batch_size 8 --learning_rate 1e-3 --lr_decay_factor 0.7 --model Attention-Unet --save_dir ./checkpoints/512/wd-2/1217_32n --n_filters 32 --output_channels 22 --device cuda:x 