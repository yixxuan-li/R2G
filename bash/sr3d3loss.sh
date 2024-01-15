#!/bin/bash
CUDA_ID=$1
LOG_NAME=$2


CUDA_VISIBLE_DEVICES=${CUDA_ID} python train.py\
        -scannet-file /data1/liyixuan/data/sr3d_scannet/keep_all_points_with_global_scan_alignment_relation_ready.pkl\
        -referit3D-file /data1/liyixuan/data/p_sr3d.csv\
        --log-dir /home/user/liyixuan/R2G/log\
        --n-workers 8\
        --batch-size 64\
        --init-lr 1e-4\
        --experiment-tag ${LOG_NAME}\
        --obj-cls-alpha 0.2\
        --target-cls-alpha 0.1\
        --anchor-cls-alpha 0.1\
        --self-supervision-alpha 0.0\
        --relation_pred False\
        --relation_retrieval False\
        --relation-cls-alpha 0.1\
        --relation_fromfile True\
        --use-GT False\
        --with_between False\
        --model-attr False\
        --multi-attr False\
        --scan-relation-path /data1/liyixuan/data/top2_relation_all.pkl\
        --relation_fromfile True\
        --use_LLM False\
        --obj-cls-path /data1/liyixuan/data/pretrained_cls.pth\

        # --resume-path /data1/liyixuan/R2G/log/testGT/01-14-2024-16-20-53/checkpoints/best_model.pth\
        # --fine-tune True
        # --mode evaluate\
        # --vocab-file /data1/liyixuan/data/vocab_nobetween.pkl\
        # --resume-path /data1/liyixuan/R2G/log/AAAI_sr3d_4loss_end2end_nobwtween/08-11-2023-21-31-51/checkpoints/best_model.pth\
        # --mode evaluate\
        # --resume-path /data1/liyixuan/R2G/log/021_1_nogt/12-26-2023-14-38-49/checkpoints/best_model.pth\
        # --obj-cls-path /data1/liyixuan/data/pretrained_cls.pth\
        