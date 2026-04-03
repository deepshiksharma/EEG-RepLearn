import os, yaml
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from models_and_co.dataset import TUHEEGHealthyAge_NPY_Dataset
from models_and_co.brain_age import Brain_Age_Predictor
from models_and_co.utils import load_pretrained_encoder, save_plots
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def main():
    set_seed(22)
    train_g, val_g, test_g = torch.Generator(), torch.Generator(), torch.Generator()
    train_g.manual_seed(22)
    val_g.manual_seed(22)
    test_g.manual_seed(22)

    # load configuration
    with open('train_(brain_age).yaml', 'r') as f:
        config = yaml.safe_load(f)

    data_config = config['data']
    model_config = config['model']
    optimizer_config = config['optimizer']
    scheduler_config = config['scheduler']
    training_config = config['training']


    # device + AMP (BF16)
    device = torch.device('cpu')
    # use_amp = device.type == 'cuda'
    # amp_dtype = torch.bfloat16
    print(device, end='\n\n')


    # datasets
    train_ds = TUHEEGHealthyAge_NPY_Dataset(
        csv_path=data_config['train_csv'],
        num_channels=data_config['num_channels'],
        T=data_config['T'],
        normalize=True
    )

    val_ds = TUHEEGHealthyAge_NPY_Dataset(
        csv_path=data_config['val_csv'],
        num_channels=data_config['num_channels'],
        T=data_config['T'],
        normalize=True
    )

    test_ds = TUHEEGHealthyAge_NPY_Dataset(
        csv_path=data_config['test_csv'],
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
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        persistent_workers=True,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        persistent_workers=True,
        pin_memory=True,
    )


    # model
    model = Brain_Age_Predictor(
        num_channels=data_config['num_channels'],
        embed_dim=model_config['embed_dim'],
        transformer_depth=model_config['transformer_depth'],
        nhead=model_config['nhead'],
        ff_dim=model_config['ff_dim'],
        dropout=model_config['dropout'],
    ).to(device)
    if model_config['pretrained_encoder']:
        load_pretrained_encoder(model, model_config['pretrained_model_path'])

    # optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config['lr']),
        betas=tuple(float(x) for x in optimizer_config['betas']),
        weight_decay=float(optimizer_config['weight_decay'])
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_config['epochs'],
        eta_min=float(scheduler_config['eta_min'])
    )

    # loss
    criterion = nn.MSELoss()


    # output
    out_dir = os.path.join('./runs', training_config['output_dir'])
    os.makedirs(out_dir, exist_ok=True)
    # save the configuration file used in this run
    with open(os.path.join(out_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)


    lowest_val_loss = float('inf')
    train_losses, val_losses, lr_values = list(), list(), list()


    # training loop
    epochs = training_config['epochs']
    for epoch in range(1, epochs + 1):

        # train
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f'train {epoch}/{epochs}', ncols=100)
        for waves, age, _ in pbar:
            waves = waves.to(device)
            age = age.to(device)

            optimizer.zero_grad()

            preds = model(waves)
            loss = criterion(preds, age)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # val
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'val {epoch}/{epochs}', ncols=100)
            for waves, age, _ in pbar:
                waves = waves.to(device)
                age = age.to(device)

                preds = model(waves)
                loss = criterion(preds, age)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step()
        lr_values.append(optimizer.param_groups[0]['lr'])

        print(f'epoch {epoch}: train={train_loss:.6f} val={val_loss:.6f}')

        # checkpointing
        last_path = os.path.join(out_dir, 'last.pth')
        torch.save(model.state_dict(), last_path)

        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            lowest_val_path = os.path.join(out_dir, f'lowest_val_loss.pth')
            torch.save(model.state_dict(), lowest_val_path)
            print(f'saved weights of the lowest val loss model from epoch {epoch}')

        save_plots(out_dir, train_losses, val_losses, lr_values)

    # test
    # load lowest val loss model
    best_path = os.path.join(out_dir, 'lowest_val_loss.pth')
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='test', ncols=100)
        for waves, age, _ in pbar:
            waves = waves.to(device)
            age = age.to(device)

            preds = model(waves)

            all_preds.append(preds.cpu())
            all_targets.append(age.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    print(f'\nTest metrics:')
    print(f'MSE: {mse:.6f}')
    print(f'MAE: {mae:.6f}')
    print(f'R2:  {r2:.6f}')


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
