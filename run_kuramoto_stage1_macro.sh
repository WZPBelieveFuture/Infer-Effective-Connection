#!/bin/sh

stage1_run_name=stage1_kuramoto_macro
time_delay=3
encoder_type=invertible
nohup python NIS_macro.py --device cuda:8 --stage1_run_name ${stage1_run_name} --data_source kuramoto --patience 10000 --train_stage 1 --epoch 10000 --time_delay ${time_delay} --encoder_type ${encoder_type} &
