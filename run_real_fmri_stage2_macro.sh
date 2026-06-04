#!/bin/sh

stage2_run_name=stage2_real_fmri_macro
stage1_run_name=stage1_real_fmri_macro
time_delay=3
encoder_type=mlp
data_path=./loc_data_real_fmri/generated_data.npz
nohup python NIS_macro.py --device cuda:1 --stage2_run_name ${stage2_run_name} --stage1_run_name ${stage1_run_name} --data_source real_fmri --data_path ${data_path} --group_schedule_mode off --causal_graph_mode none --backward_weight 0.01 --scale_id 0 --train_stage 2 --epoch 30000 --patience 30000 --time_delay ${time_delay} --encoder_type ${encoder_type} --eval_interval 500 &
nohup python NIS_macro.py --device cuda:2 --stage2_run_name ${stage2_run_name} --stage1_run_name ${stage1_run_name} --data_source real_fmri --data_path ${data_path} --group_schedule_mode off --causal_graph_mode none --backward_weight 0.01 --scale_id 1 --train_stage 2 --epoch 30000 --patience 30000 --time_delay ${time_delay} --encoder_type ${encoder_type} --eval_interval 500 &
nohup python NIS_macro.py --device cuda:2 --stage2_run_name ${stage2_run_name} --stage1_run_name ${stage1_run_name} --data_source real_fmri --data_path ${data_path} --group_schedule_mode off --causal_graph_mode auto --backward_weight 0.01 --scale_id 2 --train_stage 2 --epoch 30000 --patience 30000 --time_delay ${time_delay} --encoder_type ${encoder_type} --eval_interval 500 &
