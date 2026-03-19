import os, yaml
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import EEG_MaskedAutoencoder
from dataset import EEG_NPY_Dataset
import matplotlib.pyplot as plt


# configuration
with open('pretrain.yaml', 'r') as f:
    config = yaml.safe_load(f)

data_config = config['data']
model_config = config['model']
optimizer_config = config['optimizer']
scheduler_config = config['scheduler']
training_config = config['training']


# device + AMP (BF16)
device = torch.device('cuda')
use_amp = device.type == 'cuda'
amp_dtype = torch.bfloat16
print(device)


# datasets
train_ds = EEG_NPY_Dataset(
    csv_path=data_config['train_csv'],
    num_channels=data_config['num_channels'],
    T=data_config['T'],
    supervised=False,
)

val_ds = EEG_NPY_Dataset(
    csv_path=data_config['val_csv'],
    num_channels=data_config['num_channels'],
    T=data_config['T'],
    supervised=False,
)

# dataloaders
train_loader = DataLoader(
    train_ds,
    batch_size=data_config['batch_size'],
    shuffle=True,
    num_workers=data_config['num_workers'],
    pin_memory=True,
    drop_last=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=data_config['batch_size'],
    shuffle=False,
    num_workers=data_config['num_workers'],
    pin_memory=True,
)


# model
model = EEG_MaskedAutoencoder(
    num_channels=data_config['num_channels'],
    T=data_config['T'],
    patch_size=model_config['patch_size'],
    mask_ratio=model_config['mask_ratio'],
    embed_dim=model_config['embed_dim'],
    encoder_depth=model_config['encoder_depth'],
    decoder_depth=model_config['decoder_depth'],
    decoder_dim=model_config['decoder_dim'],
    nhead=model_config['nhead'],
    ff_dim=model_config['ff_dim'],
    conv_stem_channels=tuple(model_config['conv_stem_channels']),
    conv_decoder_hidden=model_config['conv_decoder_hidden'],
    dropout=model_config['dropout'],
).to(device)

# optimizer
optimizer = optim.AdamW(
    model.parameters(),
    lr=optimizer_config['lr'],
    betas=tuple(optimizer_config['betas']),
    weight_decay=optimizer_config['weight_decay'],
)

# scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=training_config['epochs'],
    eta_min=scheduler_config['eta_min'],
)

# loss
criterion = nn.MSELoss()


# output
out_dir = training_config['output_dir']
os.makedirs(out_dir, exist_ok=True)

best_val = float('inf')
train_hist, val_hist, lr_hist = [], [], []


# plots
def save_plots():
    # loss curve
    plt.figure()
    plt.plot(train_hist, label='train')
    plt.plot(val_hist, label='val')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.savefig(os.path.join(out_dir, 'loss.png'))
    plt.close()

    # lr curve
    plt.figure()
    plt.plot(lr_hist)
    plt.xlabel('step')
    plt.ylabel('lr')
    plt.grid()
    plt.savefig(os.path.join(out_dir, 'lr.png'))
    plt.close()


# training loop
epochs = training_config['epochs']
for epoch in range(1, epochs + 1):
    
    # train
    model.train()
    running_loss = 0.0

    pbar = tqdm(train_loader, desc=f'epoch {epoch}/{epochs}', ncols=100)

    for waves in pbar:
        waves = waves.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            decoded, target, _ = model(waves)
            loss = criterion(decoded, target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        lr_hist.append(optimizer.param_groups[0]['lr'])

        pbar.set_postfix(loss=f'{loss.item():.4f}')

    train_loss = running_loss / len(train_loader)
    train_hist.append(train_loss)

    # val
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for waves in val_loader:
            waves = waves.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                decoded, target, _ = model(waves)
                loss = criterion(decoded, target)

            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_hist.append(val_loss)

    scheduler.step()

    print(f'Epoch {epoch}: train={train_loss:.6f} val={val_loss:.6f}')

    # checkpointing
    last_path = os.path.join(out_dir, 'last_model.pth')
    torch.save(model.state_dict(), last_path)

    if val_loss < best_val:
        best_val = val_loss
        best_path = os.path.join(out_dir, 'best_model.pth')
        torch.save(model.state_dict(), best_path)

    save_plots()
