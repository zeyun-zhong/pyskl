#!/usr/bin/env bash

export MASTER_PORT=$((12000 + $RANDOM % 20000))
set -x

export CUDA_VISIBLE_DEVICES=1

torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
tools/test.py configs/mst/ntu60_xsub_3dkp/j.py \
-C work_dirs/mst/ntu60_xsub_3dkp/j/best_32x3_absrel_pos.pth --launcher pytorch
