#!/bin/bash
CUDA_ID=$1
LOG_NAME=$2


CUDA_VISIBLE_DEVICES=${CUDA_ID} python train.py\
        -scannet-file /home/yixuan/data/R2G/keep_all_points_with_global_scan_alignment_relation_ready.pkl\
        -referit3D-file /home/yixuan/data/R2G/referit3d_data/sr3d_unique.csv\
        --log-dir /home/yixuan/R2G/log\
        --n-workers 8\
        --batch-size 1\
        --init-lr 1e-4\
        --experiment-tag ${LOG_NAME}\
        --obj-cls-alpha 0.2\
        --target-cls-alpha 0.0\
        --anchor-cls-alpha 0.2\
        --self-supervision-alpha 0.0\
        --relation_pred False\
        --relation_retrieval True\
        --relation-cls-alpha 0.2\
        --use-GT True\
        --model-attr False\
        --multi-attr False\
        --obj-cls-path /home/yixuan/data/R2G/pretrained_cls.pth\
        # --resume-path /home/yixuan/R2G/log/IJCAI_nobetween/01-02-2024-08-58-44/checkpoints/best_model.pth\
        # --mode evaluate\
