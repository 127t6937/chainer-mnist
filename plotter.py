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


if __name__ == '__main__':
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
