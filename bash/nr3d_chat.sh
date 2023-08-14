#!/bin/bash
CUDA_ID=$1
LOG_NAME=$2



CUDA_VISIBLE_DEVICES=${CUDA_ID} python train.py\
        -scannet-file /home/yixuan/data/keep_all_points_00_view_with_global_scan_alignment_relation.pkl\
        -referit3D-file /home/yixuan/data/88_p_nr3d.csv\
        --log-dir /home/yixuan/R2G/log\
        --n-workers 8\
        --batch-size 52\
        --init-lr 1e-4\
        --experiment-tag ${LOG_NAME}\
        --obj-cls-alpha 0.2\
        --target-cls-alpha 0.2\
        --anchor-cls-alpha 0.0\
        --self-supervision-alpha 0.0\
        --relation_pred False\
        --relation_retrieval False\
        --relation-cls-alpha 0.0\
        --use-GT False\
        --model-attr True\
        --multi-attr True\
        --use_LLM True\
        --scan-relation-path /home/yixuan/data/top2_relation_all.pkl\
        # --resume-path /home/yixuan/R2G/log/dot/08-14-2023-02-19-39/checkpoints/best_model.pth\
        # --mode evaluate
        # --obj-cls-path /home/yixuan/data/pretrained_cls.pth