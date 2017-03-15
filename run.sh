# for op in SGD MomentumSGD NesterovAG AdaGrad AdaDelta RMSprop Adam; do
#     python train_mnist.py --gpu 0 --optimizer $op --out result_$op
# done

for unit in 32 64 128 256 512 1024 2048 4096; do
    python train_mnist.py --gpu 0 --unit $unit --out result_unit_$unit
done
