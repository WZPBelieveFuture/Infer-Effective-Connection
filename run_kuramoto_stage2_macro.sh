#!/bin/sh

stage2_run_name=stage2_kuramoto_macro
stage1_run_name=stage1_kuramoto_macro
time_delay=3
encoder_type=mlp 
nohup python NIS_macro.py --device cuda:0 --scale_id 0 --stage2_run_name ${stage2_run_name} --stage1_run_name ${stage1_run_name} --backward_weight 0.01 --data_source kuramoto --train_stage 2 --epoch 30000 --patience 20000 --time_delay ${time_delay} --encoder_type ${encoder_type} --eval_interval 500 &
nohup python NIS_macro.py --device cuda:4 --scale_id 1 --stage2_run_name ${stage2_run_name} --stage1_run_name ${stage1_run_name} --backward_weight 0.01 --data_source kuramoto --train_stage 2 --epoch 30000 --patience 20000 --time_delay ${time_delay} --encoder_type ${encoder_type} --eval_interval 500 &
nohup python NIS_macro.py --device cuda:3 --scale_id 2 --stage2_run_name ${stage2_run_name} --stage1_run_name ${stage1_run_name} --backward_weight 0.01 --data_source kuramoto --train_stage 2 --epoch 30000 --patience 20000 --time_delay ${time_delay} --encoder_type ${encoder_type} --eval_interval 500 &
nohup python NIS_macro.py --device cuda:1 --scale_id 3 --stage2_run_name ${stage2_run_name} --stage1_run_name ${stage1_run_name} --backward_weight 0.01 --data_source kuramoto --train_stage 2 --epoch 30000 --patience 20000 --time_delay ${time_delay} --encoder_type ${encoder_type} --eval_interval 500 &
