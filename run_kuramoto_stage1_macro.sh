#!/bin/sh

stage1_run_name=stage1_kuramoto_macro
time_delay=1
encoder_type=mlp
nohup python NIS_macro.py --device cuda:1 --stage1_run_name ${stage1_run_name} --data_source kuramoto --patience 20000 --train_stage 1 --epoch 20000 --time_delay ${time_delay} --encoder_type ${encoder_type} &
