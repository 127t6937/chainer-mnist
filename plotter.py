import matplotlib.pyplot as plt
import json

def plot_result(logfiles, target='validation/main/accuracy', outfile=None):
    if not target in ['main/loss', 'main/accuracy',
                      'validation/main/loss', 'validation/main/accuracy']:
        print('invalid target: {}'.format(target))
        exit(1)

    fig, ax = plt.subplots()

    for label, logfile in logfiles:
        result = json.load(open(logfile))
        epoch = []
        loss = []
        for x in result:
            epoch.append(x['epoch'])
            loss.append(x[target])
        ax.plot(epoch, loss, label=label, marker='.')

    ax.set_xlabel('epoch')
    ax.set_ylabel(target)
    ax.legend(loc='best')
    ax.grid(True)
    fig.tight_layout()

    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()


def opt():
    logfiles = [('SGD', 'result_SGD/log'),
                ('MomentumSGD', 'result_MomentumSGD/log'),
                ('NesterovAG', 'result_NesterovAG/log'),
                ('AdaGrad', 'result_AdaGrad/log'),
                ('AdaDelta', 'result_AdaDelta/log'),
                ('RMSprop', 'result_RMSprop/log'),
                ('Adam', 'result_Adam/log')]

    plot_result(logfiles, 'main/loss', 'opt_loss.png')
    plot_result(logfiles, 'main/accuracy', 'opt_acc.png')
    plot_result(logfiles, 'validation/main/loss', 'opt_val_loss.png')
    plot_result(logfiles, 'validation/main/accuracy', 'opt_val_acc.png')


def unit():
    logfiles = [('32', 'result_unit_32/log'),
                ('64', 'result_unit_64/log'),
                ('128', 'result_unit_128/log'),
                ('256', 'result_unit_256/log'),
                ('512', 'result_unit_512/log'),
                ('1024', 'result_unit_1024/log'),
                ('2048', 'result_unit_2048/log'),
                ('4096', 'result_unit_4096/log')]

    plot_result(logfiles, 'main/loss', 'unit_loss.png')
    plot_result(logfiles, 'main/accuracy', 'unit_acc.png')
    plot_result(logfiles, 'validation/main/loss', 'unit_val_loss.png')
    plot_result(logfiles, 'validation/main/accuracy', 'unit_val_acc.png')


def mlp():
    logfiles = [('#hidden=1', 'result_MLP1/log'),
                ('#hidden=2', 'result_MLP2/log'),
                ('#hidden=3', 'result_MLP3/log'),
                ('#hidden=4', 'result_MLP4/log'),
                ('#hidden=5', 'result_MLP5/log')]

    plot_result(logfiles, 'main/loss', 'mlp_loss.png')
    plot_result(logfiles, 'main/accuracy', 'mlp_acc.png')
    plot_result(logfiles, 'validation/main/loss', 'mlp_val_loss.png')
    plot_result(logfiles, 'validation/main/accuracy', 'mlp_val_acc.png')


def activation():
    logfiles = [('sigmoid', 'result_sigmoid/log'),
                ('tanh', 'result_tanh/log'),
                ('relu', 'result_relu/log'),
                ('leaky_relu', 'result_leaky_relu/log'),
                ('elu', 'result_elu/log')]

    plot_result(logfiles, 'main/loss', 'act_loss.png')
    plot_result(logfiles, 'main/accuracy', 'act_acc.png')
    plot_result(logfiles, 'validation/main/loss', 'act_val_loss.png')
    plot_result(logfiles, 'validation/main/accuracy', 'act_val_acc.png')


def cnn():
    logfiles = [('CNN1', 'result_CNN1/log'),
                ('CNN2', 'result_CNN2/log'),
                ('CNN3', 'result_CNN3/log'),
                ('MLP3', 'result_MLP3/log')]

    plot_result(logfiles, 'main/loss', 'cnn_loss.png')
    plot_result(logfiles, 'main/accuracy', 'cnn_acc.png')
    plot_result(logfiles, 'validation/main/loss', 'cnn_val_loss.png')
    plot_result(logfiles, 'validation/main/accuracy', 'cnn_val_acc.png')


if __name__ == '__main__':
    # opt()
    # unit()
    # mlp()
    # activation()
    cnn()
