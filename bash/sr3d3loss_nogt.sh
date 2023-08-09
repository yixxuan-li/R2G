#!/bin/bash
CUDA_ID=$1
LOG_NAME=$2


CUDA_VISIBLE_DEVICES=${CUDA_ID} python train.py\
        -scannet-file /home/yixuan/data/keep_all_points_with_global_scan_alignment_relation_ready.pkl\
        -referit3D-file /home/yixuan/data/sr3d.csv\
        --log-dir /home/yixuan/R2G/log\
        --n-workers 8\
        --batch-size 64\
        --init-lr 1e-6\
        --experiment-tag ${LOG_NAME}\
        --obj-cls-alpha 0.2\
        --target-cls-alpha 0.2\
        --anchor-cls-alpha 0.2\
        --self-supervision-alpha 0.0\
        --relation_pred False\
        --relation_retrieval False\
        --relation-cls-alpha 0.2\
        --use-GT False\
        --model-attr False\
        --multi-attr False\
        --scan-relation-path /home/yixuan/data/top2_relation_all.pkl\
        --resume-path /home/yixuan/R2G/log/ft/08-09-2023-21-18-23/checkpoints/best_model.pth\
        --obj-cls-path /home/yixuan/data/pretrained_cls.pth\
        --mode evaluate\