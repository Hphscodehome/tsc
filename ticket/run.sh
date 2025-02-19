#!/bin/bash
#num_runs=13
#for i in $(seq 3 $num_runs); do
#9 13 17 21 25 29 31
for i in 4; do
  echo "运行第 $i 次："
  /root/miniconda3/envs/sumo/bin/python3 get_issue_values.py > /data/hupenghui/Self/tsc/ticket/get_issue_values.log 2>&1 
  /root/miniconda3/envs/sumo/bin/python3 get_statistics.py > /data/hupenghui/Self/tsc/ticket/get_statistics.log 2>&1 
  /root/miniconda3/envs/sumo/bin/python3 make_train_data_logits.py --end $i > /data/hupenghui/Self/tsc/ticket/logits/make_train_data_logits.log 2>&1 
  /root/miniconda3/envs/sumo/bin/python3 model_train.py --end $i > /data/hupenghui/Self/tsc/ticket/logits/model_train.log 2>&1 
  /root/miniconda3/envs/sumo/bin/python3 model_eval.py  --end $i > /data/hupenghui/Self/tsc/ticket/logits/model_eval.log 2>&1 
  /root/miniconda3/envs/sumo/bin/python3 make_train_data_arti.py --end $i > /data/hupenghui/Self/tsc/ticket/arti/make_train_data_arti.log 2>&1 
  /root/miniconda3/envs/sumo/bin/python3 model_train_v2.py --end $i > /data/hupenghui/Self/tsc/ticket/arti/model_train_v2.log 2>&1 
  /root/miniconda3/envs/sumo/bin/python3 model_eval_v2.py --end $i > /data/hupenghui/Self/tsc/ticket/arti/model_eval_v2.log 2>&1 
  echo "第 $i 次运行结束。"
done
echo "所有运行完成。"
