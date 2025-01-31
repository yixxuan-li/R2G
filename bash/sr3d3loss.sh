#!/bin/bash
CUDA_ID=$1
LOG_NAME=$2


CUDA_VISIBLE_DEVICES=${CUDA_ID} python train.py\
        -scannet-file /home/yixuan/data/R2G/keep_all_points_with_global_scan_alignment_relation_ready.pkl\
        -referit3D-file /home/yixuan/data/R2G/referit3d_data/p_sr3d.csv\
        --log-dir /home/yixuan/R2G/log\
        --n-workers 8\
        --batch-size 64\
        --init-lr 1e-5\
        --experiment-tag ${LOG_NAME}\
        --obj-cls-alpha 0.2\
        --target-cls-alpha 0.2\
        --anchor-cls-alpha 0.2\
        --self-supervision-alpha 0.0\
        --relation_pred False\
        --relation_retrieval False\
        --relation-cls-alpha 0.2\
        --use-GT False\
        --with_between True\
        --model-attr False\
        --multi-attr False\
        --scan-relation-path /home/yixuan/data/R2G/top2_relation_all.pkl\
        --relation_fromfile True\
        --edge_onehot True\
        --use_LLM False\
        --resume-path /home/yixuan/R2G/log/finetune_allloss_between/01-17-2024-15-30-28/checkpoints/best_model.pth\
        --mode evaluate\
        # --obj-cls-path /home/yixuan/data/R2G/pretrained_cls.pth\
        # --vocab-file /data1/liyixuan/data/vocab_nobetween.pkl\
        # --resume-path /data1/liyixuan/R2G/log/AAAI_sr3d_4loss_end2end_nobwtween/08-11-2023-21-31-51/checkpoints/best_model.pth\
        # --mode evaluate\
        # --resume-path /data1/liyixuan/R2G/log/021_1_nogt/12-26-2023-14-38-49/checkpoints/best_model.pth\
        # --obj-cls-path /data1/liyixuan/data/pretrained_cls.pth\
        