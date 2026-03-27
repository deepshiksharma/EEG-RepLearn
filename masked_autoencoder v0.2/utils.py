import os, random
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed=22):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to: {seed}")

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def save_plots_and_loss_arrays(out_dir, train_loss_list, train_debug_loss_list,
                               val_loss_list, lr_value_list, plot_train_debug_loss=False):
    # loss curve
    plt.figure()
    plt.plot(train_loss_list, label='train', color='blue')
    if plot_train_debug_loss:
        plt.plot(train_debug_loss_list, label='train_debug', color='green')
    plt.plot(val_loss_list, label='val', color='orange')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.savefig(os.path.join(out_dir, 'loss.png'), dpi=300)
    plt.close()

    # lr curve
    plt.figure()
    plt.plot(lr_value_list)
    plt.xlabel('step')
    plt.ylabel('lr')
    plt.grid()
    plt.savefig(os.path.join(out_dir, 'lr.png'), dpi=300)
    plt.close()

    train_loss_list = np.array(train_loss_list)
    val_loss_list = np.array(val_loss_list)
    np.save(os.path.join(out_dir, 'train_loss_list.npy'), train_loss_list)
    np.save(os.path.join(out_dir, 'val_loss_list.npy'), val_loss_list)
    if plot_train_debug_loss:
        train_debug_loss_list = np.array(train_debug_loss_list)
        np.save(os.path.join(out_dir, 'train_debug_loss_list.npy'), train_debug_loss_list)


def save_recon(out_dir, epoch, input_wave, recon_wave, mask_tokens, channel_idx=0):
    """
    input_wave: torch.Tensor or np.ndarray, shape (C, T)
    recon_wave: torch.Tensor or np.ndarray, shape (C, T)
    mask_tokens: torch.Tensor or np.ndarray, shape (N,)
                 True = masked / reconstructed region
    """

    if hasattr(input_wave, "detach"):
        input_wave = input_wave.detach().cpu().float().numpy()
    if hasattr(recon_wave, "detach"):
        recon_wave = recon_wave.detach().cpu().float().numpy()
    if hasattr(mask_tokens, "detach"):
        mask_tokens = mask_tokens.detach().cpu().bool().numpy()

    T = input_wave.shape[-1]
    N = mask_tokens.shape[0]
    chunk_size = T // N

    # token-level mask -> time-level mask
    time_mask = np.repeat(mask_tokens, chunk_size)

    # save npy files
    np.save(os.path.join(out_dir, f"epoch_{epoch}_input.npy"), input_wave)
    np.save(os.path.join(out_dir, f"epoch_{epoch}_recon.npy"), recon_wave)
    np.save(os.path.join(out_dir, f"epoch_{epoch}_mask_tokens.npy"), mask_tokens)
    np.save(os.path.join(out_dir, f"epoch_{epoch}_mask_time.npy"), time_mask)
    
    # plot one channel
    ch = int(channel_idx)
    x = np.arange(T)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(x, input_wave[ch], label="input", linewidth=1)
    ax.plot(x, recon_wave[ch], label="recon", linewidth=1, alpha=0.8)

    # shade masked regions
    in_span = False
    start = 0
    for i, m in enumerate(time_mask):
        if m and not in_span:
            start = i
            in_span = True
        elif not m and in_span:
            ax.axvspan(start, i, alpha=0.15)
            in_span = False
    if in_span:
        ax.axvspan(start, T, alpha=0.15)

    ax.set_title(f"Epoch {epoch} | channel {ch}")
    ax.set_xlabel("time")
    ax.set_ylabel("amplitude")
    ax.legend(loc="upper right")
    fig.tight_layout()

    fig_path = os.path.join(out_dir, f"epoch_{epoch}_ch{ch}.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
