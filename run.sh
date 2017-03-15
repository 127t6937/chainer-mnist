#!/usr/bin/env bash

# optimizer test
for op in SGD MomentumSGD NesterovAG AdaGrad AdaDelta RMSprop Adam; do
    python train_mnist.py --gpu 0 --optimizer $op --out result_$op
done

# hidden unit test
for unit in 32 64 128 256 512 1024 2048 4096; do
    python train_mnist.py --gpu 0 --unit $unit --out result_unit_$unit
done

# hidden layer test
for model in MLP1 MLP2 MLP3 MLP4 MLP5; do
    python train_mnist.py --gpu 0 --model $model --out result_$model
done

# activation test
for activation in sigmoid tanh relu leaky_relu elu; do
    python train_mnist.py --gpu 0 --activation $activation --out result_$activation
done

# cnn test
for model in CNN1 CNN2 CNN3; do
    python train_mnist.py --gpu 0 --model $model --out result_$model
done
