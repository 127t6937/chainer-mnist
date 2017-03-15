#!/usr/bin/env bash

# # optimizer test
# for op in SGD MomentumSGD NesterovAG AdaGrad AdaDelta RMSprop Adam; do
#     python train_mnist.py --gpu 0 --optimizer $op --out result_$op
# done

# # hidden unit test
# for unit in 32 64 128 256 512 1024 2048 4096; do
#     python train_mnist.py --gpu 0 --unit $unit --out result_unit_$unit
# done

for model in MLP1 MLP2 MLP3 MLP4 MLP5; do
    python train_mnist.py --gpu 0 --model $model --out result_$model
done
