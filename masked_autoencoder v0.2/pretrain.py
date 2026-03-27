import os, time
os.environ['PYTHONHASHSEED'] = '22'
import yaml
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from model import EEG_MaskedAutoencoder
from dataset import TUHEEGHealthy_NPY_Dataset
from utils import save_plots_and_loss_arrays, set_seed, seed_worker, save_recon


def main():
    set_seed(22)
    train_g, val_g = torch.Generator(), torch.Generator()
    train_g.manual_seed(22)
    val_g.manual_seed(22)

    # load configuration
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
    print(device, end='\n\n')


    # datasets
    train_ds = TUHEEGHealthy_NPY_Dataset(
        csv_path=data_config['train_csv'],
        num_channels=data_config['num_channels'],
        T=data_config['T'],
        normalize=True
    )

    val_ds = TUHEEGHealthy_NPY_Dataset(
        csv_path=data_config['val_csv'],
        num_channels=data_config['num_channels'],
        T=data_config['T'],
        normalize=True
    )

    # dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=train_g,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=val_g,
        pin_memory=True
    )


    # model
    model = EEG_MaskedAutoencoder(
        num_channels=data_config['num_channels'],
        T=data_config['T'],
        mask_ratio=model_config['mask_ratio'],
        embed_dim=model_config['embed_dim'],
        transformer_depth=model_config['transformer_depth'],
        nhead=model_config['nhead'],
        ff_dim=model_config['ff_dim'],
        conv_decoder_hidden=model_config['conv_decoder_hidden'],
        dropout=model_config['dropout']
    ).to(device)

    # optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config['lr']),
        betas=tuple(float(x) for x in optimizer_config['betas']),
        weight_decay=float(optimizer_config['weight_decay'])
    )

    # scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_config['epochs'],
        eta_min=float(scheduler_config['eta_min'])
    )

    # loss
    criterion = nn.MSELoss()



    save_recon_every_n_epochs = training_config['save_recon_every_n_epochs']

    # output directory
    base_output_dir = "./runs"
    os.makedirs(base_output_dir, exist_ok=True)

    out_dir = os.path.join(base_output_dir, training_config['output_dir'])
    os.mkdir(out_dir)

    recon_dir = os.path.join(out_dir, 'reconstructions')
    weights_dir = os.path.join(out_dir, 'weights')
    os.mkdir(recon_dir)
    os.mkdir(weights_dir)

    # save the configuration file used in this run
    with open(os.path.join(out_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)


    lowest_val_loss = float('inf')
    train_losses, val_losses, lr_values = list(), list(), list()
    train_debug_losses = list()


    # training loop
    epochs = training_config['epochs']
    for epoch in range(1, epochs + 1):
        
        # train
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f'train epoch {epoch}/{epochs}', ncols=100)
        for waves in pbar:
            waves = waves.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                decoded, target, _, _ = model(waves)
                loss = criterion(decoded, target)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # run model.eval() on the train set
        # (to debug the issue where val loss is consistently lower than train loss)
        model.eval()
        train_debug_loss = 0.0

        with torch.no_grad():
            pbar = tqdm(train_loader, desc=f'train_debug epoch {epoch}/{epochs}', ncols=100)
            for waves in pbar:
                waves = waves.to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    decoded, target, _, _ = model(waves)
                    loss = criterion(decoded, target)
                
                train_debug_loss += loss.item()

        train_debug_loss /= len(train_loader)
        train_debug_losses.append(train_debug_loss)
        
        # val
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'val epoch {epoch}/{epochs}', ncols=100)
            for batch_idx, waves in enumerate(pbar):
                waves = waves.to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    decoded, target, mask, recon = model(waves)
                    loss = criterion(decoded, target)

                val_loss += loss.item()
                
                if ((epoch % save_recon_every_n_epochs == 0) or epoch == 1) and batch_idx == 0:
                    save_recon(out_dir=recon_dir, epoch=epoch,
                               input_wave=waves[0], recon_wave=recon[0], mask_tokens=mask[0], channel_idx=0)

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step()
        lr_values.append(optimizer.param_groups[0]['lr'])

        print(f'Epoch {epoch}: train={train_loss:.6f} train_debug={train_debug_loss:.6f} val={val_loss:.6f}')

        # checkpointing
        last_path = os.path.join(weights_dir, f'epoch_{epoch}.pth')
        torch.save(model.state_dict(), last_path)

        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            print(f'lowest val loss yet at epoch {epoch}')

        save_plots_and_loss_arrays(out_dir, train_losses, train_debug_losses,
                                   val_losses, lr_values, plot_train_debug_loss=True)


if __name__ == "__main__":
    start_time = time.perf_counter()

    mp.set_start_method("spawn", force=True)
    main()

    end_time = time.perf_counter()
    time_in_mins = (end_time-start_time)/60
    print(f'\nthis run took {time_in_mins:.1f} minutes to complete.')
