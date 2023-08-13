#!/bin/bash
# --resume-path /data1/liyixuan/R2G/log/obj_cls/07-27-2023-17-08-36/checkpoints/best_model.pth\
CUDA_ID=$1
LOG_NAME=$2


CUDA_VISIBLE_DEVICES=${CUDA_ID} python train.py\
        -scannet-file /data1/liyixuan/data/sr3d_scannet/keep_all_points_with_global_scan_alignment_relation_ready.pkl\
        -referit3D-file /data1/liyixuan/data/p_sr3d.csv\
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
        --model-attr False\
        --multi-attr False\
        --scan-relation-path /data1/liyixuan/data/top2_relation_all.pkl\
        --use_LLM True
        # --obj-cls-path /data1/liyixuan/data/pretrained_cls.pth\
        # --vocab-file /data1/liyixuan/data/sr3d_vocab.pkl
                # --fine-tune True\

