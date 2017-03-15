for op in SGD MomentumSGD NesterovAG AdaGrad AdaDelta RMSprop Adam; do
    python train_mnist.py --gpu 0 --optimizer $op --out result_$op
done
