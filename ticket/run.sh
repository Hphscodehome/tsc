#!/bin/bash
#num_runs=13
#for i in $(seq 3 $num_runs); do
for i in 13 17 21 25 31; do
  echo "运行第 $i 次："
  /root/miniconda3/envs/sumo/bin/python3 construct_model.py --end $i >> train.log 2>&1 
  echo "第 $i 次运行结束。"
done
echo "所有运行完成。"