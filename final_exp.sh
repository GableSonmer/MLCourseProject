#!/bin/bash

# 定义参数
pred_lens=("96" "336")
optimizers=("sgd" "adam" "rmsprop")
lrs=("0.01" "0.001" "0.0001" "0.00001")
bss=("32" "64" "128" "256" "512")

# 遍历所有参数组合
# shellcheck disable=SC2068
for pred_len in ${pred_lens[@]}; do
  for optimizer in ${optimizers[@]}; do
    for lr in ${lrs[@]}; do
      for bs in ${bss[@]}; do
        # 执行命令
        echo "Running with pred_len=$pred_len, optimizer=$optimizer, lr=$lr, bs=$bs"
        python -u main.py --model transformer --epochs 500 --pred_len $pred_len --optimizer $optimizer --lr $lr --batch_size $bs
#        python -u main.py --model lstm --epochs 500 --pred_len $pred_len --optimizer $optimizer --lr $lr --batch_size $bs
      done
    done
  done
done

#python -u main.py --model transformer --epochs 1 --pred_len 96 --optimizer sgd --lr 0.01 --batch_size 32 --iteration 1