import os
import matplotlib.pyplot as plt


def save_plots(out_dir, train_loss_list, val_loss_list, lr_value_list):
    # loss curve
    plt.figure()
    plt.plot(train_loss_list, label='train')
    plt.plot(val_loss_list, label='val')
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
