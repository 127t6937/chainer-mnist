import argparse

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions

from net import *


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--optimizer', '-op',
                        choices=('SGD', 'MomentumSGD', 'NesterovAG', 'AdaGrad',
                                 'AdaDelta', 'RMSprop', 'Adam'),
                        default='SGD', help='optimization type')
    parser.add_argument('--model', '-m',
                        choices=('MLP1', 'MLP2', 'MLP3', 'MLP4', 'MLP5',
                                 'CNN1', 'CNN2', 'CNN3'),
                        default='MLP3', help='model type')
    parser.add_argument('--activation', '-a',
                        choices=('sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu'),
                        default='relu')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('optimizer: {}'.format(args.optimizer))
    print('model type: {}'.format(args.model))
    print('activation: {}'.format(args.activation))
    print('')

    # Set up a neural network to train
    if args.activation == 'sigmoid':
        activation = F.sigmoid
    elif args.activation == 'tanh':
        activation = F.tanh
    elif args.activation == 'relu':
        activation = F.relu
    elif args.activation == 'leaky_relu':
        activation = F.leaky_relu
    elif args.activation == 'elu':
        activation = F.elu

    if args.model == 'MLP1':
        model = MLP1(args.unit, 10, activation)
    elif args.model == 'MLP2':
        model = MLP2(args.unit, 10, activation)
    elif args.model == 'MLP3':
        model = MLP3(args.unit, 10, activation)
    elif args.model == 'MLP4':
        model = MLP4(args.unit, 10, activation)
    elif args.model == 'MLP5':
        model = MLP5(args.unit, 10, activation)
    elif args.model == 'CNN1':
        model = CNN1(10)
    elif args.model == 'CNN2':
        model = CNN2(10)
    elif args.model == 'CNN3':
        model = CNN3(10)

    model = L.Classifier(model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    # Setup an optimizer
    if args.optimizer == 'SGD':
        optimizer = chainer.optimizers.SGD()
    elif args.optimizer == 'MomentumSGD':
        optimizer = chainer.optimizers.MomentumSGD()
    elif args.optimizer == 'NesterovAG':
        optimizer = chainer.optimizers.NesterovAG()
    elif args.optimizer == 'AdaGrad':
        optimizer = chainer.optimizers.AdaGrad()
    elif args.optimizer == 'AdaDelta':
        optimizer = chainer.optimizers.AdaDelta()
    elif args.optimizer == 'RMSprop':
        optimizer = chainer.optimizers.RMSprop()
    elif args.optimizer == 'Adam':
        optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load the MNIST dataset
    if args.model.startswith('CNN'):
        train, test = chainer.datasets.get_mnist(ndim=3)
    else:
        train, test = chainer.datasets.get_mnist()

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'validation/main/loss'],
        'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(
        ['main/accuracy', 'validation/main/accuracy'],
        'epoch', file_name='accuracy.png'))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
