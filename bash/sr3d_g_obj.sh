#!/bin/bash
CUDA_ID=$0
LOG_NAME=$1

CUDA_VISIBLE_DEVICES=2
python train.py\
        -scannet-file /data1/liyixuan/data/sr3d_scannet/keep_all_points_with_global_scan_alignment_relation_ready.pkl\
        -referit3D-file /data1/liyixuan/data/sr3d/sr3d.csv\
        --log-dir /home/user/liyixuan/R2G/log\
        --n-workers 8\
        --batch-size 1\
        --init-lr 1e-4\
        --experiment-tag ${LOG_NAME}\
        --obj-cls-alpha 0.2\
        --target-cls-alpha 0.0\
        --anchor-cls-alpha 0.0\
        --self-supervision-alpha 0.0\
        --relation_pred False\
        --relation_retrieval True\
        --relation-cls-alpha 0.0\
        --use-GT True\
        --model-attr False\
        --multi-attr False



CUDA_VISIBLE_DEVICES=1 python train.py --mode evaluate -scannet-file /data1/liyixuan/data/sr3d_scannet/keep_all_points_with_global_scan_alignment_relation_ready.pkl -referit3D-file /data1/liyixuan/data/sr3d/sr3d_unique_2.csv --log-dir /home/user/liyixuan/R2G/log --n-workers 8 --batch-size 1 --init-lr 1e-4 --obj-cls-alpha 0.2 --target-cls-alpha 0.2 --anchor-cls-alpha 0.2 --self-supervision-alpha 0.0 --relation_pred False --relation_retrieval True --relation-cls-alpha 0.2 --use-GT False --model-attr False --multi-attr False --resume-path /data1/liyixuan/R2G/log/final_sr3d_nogt/05-12-2023-00-58-26/checkpoints/best_model.pth  --vocab-file /data1/liyixuan/data/sr3d_vocab.pkl  --experiment-tag test




--resume-path /data1/liyixuan/R2G/log/sr3d3loss_nogt/07-24-2023-23-32-13/checkpoints/best_model.pth --obj-cls-path /data1/liyixuan/R2G/log/final_sr3d_nogt/05-12-2023-00-58-26/checkpoints/best_model.pth