#!/bin/bash
CUDA_ID=$0
LOG_NAME=$1

CUDA_VISIBLE_DEVICES=0
python train.py\
        -scannet-file /data1/liyixuan/data/sr3d_scannet/keep_all_points_with_global_scan_alignment.pkl\
        -referit3D-file /data1/liyixuan/data/sr3d/sr3d.csv\
        --log-dir /home/user/liyixuan/R2G/log\
        --n-workers 8\
        --batch-size 64\
        --init-lr 1e-4\
        --experiment-tag ${LOG_NAME}\
        --obj-cls-alpha 0.0\
        --target-cls-alpha 0.2\
        --anchor-cls-alpha 0.2\
        --self-supervision-alpha 0.0\
        --relation_pred False\
        --relation_retrieval True\
        --relation-cls-alpha 0.2\
        --use-GT True\
        --model-attr False\
        --multi-attr False