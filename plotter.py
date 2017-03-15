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


if __name__ == '__main__':
    # opt()
    unit()
