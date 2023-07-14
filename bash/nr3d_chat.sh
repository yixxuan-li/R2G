#!/bin/bash
LOG_NAME=$1



CUDA_VISIBLE_DEVICES=0 python train.py\
        -scannet-file /data1/liyixuan/data/keep_all_points_00_view_with_global_scan_alignment_relation_ready.pkl\
        -referit3D-file /data1/liyixuan/data/p_nr3d.csv\
        --log-dir /home/user/liyixuan/R2G/log\
        --n-workers 8\
        --batch-size 32\
        --init-lr 1e-4\
        --experiment-tag ${LOG_NAME}\
        --obj-cls-alpha 0.0\
        --target-cls-alpha 0.0\
        --anchor-cls-alpha 0.0\
        --self-supervision-alpha 0.0\
        --relation_pred False\
        --relation_retrieval True\
        --relation-cls-alpha 0.0\
        --use-GT True\
        --model-attr True\
        --multi-attr False\
        --use_LLM True