import os
import matplotlib.pyplot as plt


def save_plots(out_dir, train_loss_list, train_debug_losses, val_loss_list, lr_value_list):
    # loss curve
    plt.figure()
    plt.plot(train_loss_list, label='train', color='blue')
    plt.plot(train_debug_losses, label='train', color='green')
    plt.plot(val_loss_list, label='val', color='orange')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.savefig(os.path.join(out_dir, 'loss.png'))
    plt.close()

    # lr curve
    plt.figure()
    plt.plot(lr_value_list)
    plt.xlabel('step')
    plt.ylabel('lr')
    plt.grid()
    plt.savefig(os.path.join(out_dir, 'lr.png'))
    plt.close()
