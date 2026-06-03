#!/bin/sh

stage2_run_name=stage2_kuramoto_macro
stage1_run_name=stage1_kuramoto_macro
time_delay=3
encoder_type=invertible 
nohup python NIS_macro.py --device cuda:9 --scale_id 0 --stage2_run_name ${stage2_run_name} --stage1_run_name ${stage1_run_name} --backward_weight 1 --data_source kuramoto --train_stage 2 --epoch 30000 --patience 20000 --time_delay ${time_delay} --encoder_type ${encoder_type} --eval_interval 500 &
nohup python NIS_macro.py --device cuda:7 --scale_id 1 --stage2_run_name ${stage2_run_name} --stage1_run_name ${stage1_run_name} --backward_weight 1 --data_source kuramoto --train_stage 2 --epoch 30000 --patience 20000 --time_delay ${time_delay} --encoder_type ${encoder_type} --eval_interval 500 &
nohup python NIS_macro.py --device cuda:6 --scale_id 2 --stage2_run_name ${stage2_run_name} --stage1_run_name ${stage1_run_name} --backward_weight 1 --data_source kuramoto --train_stage 2 --epoch 30000 --patience 20000 --time_delay ${time_delay} --encoder_type ${encoder_type} --eval_interval 500 &
nohup python NIS_macro.py --device cuda:8 --scale_id 3 --stage2_run_name ${stage2_run_name} --stage1_run_name ${stage1_run_name} --backward_weight 1 --data_source kuramoto --train_stage 2 --epoch 30000 --patience 20000 --time_delay ${time_delay} --encoder_type ${encoder_type} --eval_interval 500 &
