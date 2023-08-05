#!/bin/bash
CUDA_ID=$1
LOG_NAME=$2


CUDA_VISIBLE_DEVICES=${CUDA_ID} python train.py\
        -scannet-file /data1/liyixuan/data/sr3d_scannet/keep_all_points_with_global_scan_alignment_relation_ready.pkl\
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
        --relation_retrieval False\
        --relation-cls-alpha 0.2\
        --use-GT False\
        --model-attr False\
        --multi-attr False\
        --fine-tune True\
        --resume-path /data1/liyixuan/R2G/log/obj_cls/07-27-2023-17-08-36/checkpoints/best_model.pth\
        --scan-relation-path /data1/liyixuan/data/top2_relation_all.pkl