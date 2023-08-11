#!/bin/bash
CUDA_ID=$1
LOG_NAME=$2

CUDA_VISIBLE_DEVICES=${CUDA_ID} python train.py\
        -scannet-file /data1/liyixuan/data/keep_all_points_00_view_with_global_scan_alignment_relation_ready.pkl\
        -referit3D-file /data1/liyixuan/referit_my/referit3d/data/language/nr3d/nr3d.csv\
        --log-dir /home/user/liyixuan/R2G/log\
        --n-workers 8\
        --batch-size 64\
        --init-lr 5e-4\
        --experiment-tag ${LOG_NAME}\
        --obj-cls-alpha 0.2\
        --target-cls-alpha 0.0\
        --anchor-cls-alpha 0.0\
        --self-supervision-alpha 0.0\
        --relation_pred False\
        --relation_retrieval False\
        --relation-cls-alpha 0.0\
        --use-GT False\
        --model-attr True\
        --multi-attr True\
        --scan-relation-path /data1/liyixuan/data/top2_relation_all.pkl\
        # --scan-relation-path /data1/liyixuan/data/top2_relation_all.pkl\