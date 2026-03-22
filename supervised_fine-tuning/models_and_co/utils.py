import os, torch
import matplotlib.pyplot as plt


def load_pretrained_encoder(model, path):
    state = torch.load(path, map_location='cpu')
    model_dict = model.state_dict()

    # filter only encoder weights
    pretrained = {k: v for k, v in state.items() if k.startswith('encoder.')}
    model_dict.update(pretrained)

    model.load_state_dict(model_dict)
    print('pretrained encoder loaded', end='\n\n')


def save_plots(out_dir, train_loss_list, val_loss_list, lr_value_list):
    # loss curve
    plt.figure()
    plt.plot(train_loss_list, label='train')
    plt.plot(val_loss_list, label='val')
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
