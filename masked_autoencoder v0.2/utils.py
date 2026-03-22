import os
import numpy as np
import matplotlib.pyplot as plt


def save_plots_and_loss_arrays(out_dir, train_loss_list, train_debug_loss_list,
                               val_loss_list, lr_value_list, plot_train_debug_loss=False):
    # loss curve
    plt.figure()
    plt.plot(train_loss_list, label='train', color='blue')
    if plot_train_debug_loss:
        plt.plot(train_debug_loss_list, label='train', color='green')
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
    train_debug_loss_list = np.array(train_debug_loss_list)
    val_loss_list = np.array(val_loss_list)
    np.save(os.path.join(out_dir, 'train_loss_list.npy'), train_loss_list)
    np.save(os.path.join(out_dir, 'train_debug_loss_list.npy'), train_debug_loss_list)
    np.save(os.path.join(out_dir, 'val_loss_list.npy'), val_loss_list)
