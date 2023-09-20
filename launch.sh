#!/usr/bin/env bash

export MASTER_PORT=$((12000 + $RANDOM % 20000))
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 --master_port=$MASTER_PORT \
tools/train.py configs/lst/ntu60_xsub_3dkp/j_vanilla_variable_dim_mask_pool.py --launcher pytorch --validate --seed 42


torchrun --nproc_per_node=4 --master_port=$MASTER_PORT \
tools/train.py configs/lst/ntu60_xsub_3dkp/j_vanilla_variable_dim.py --launcher pytorch --validate --seed 42


torchrun --nproc_per_node=4 --master_port=$MASTER_PORT \
tools/train.py configs/lst/ntu60_xsub_3dkp/j_vanilla_variable_dim_mask.py --launcher pytorch --validate --seed 42


torchrun --nproc_per_node=4 --master_port=$MASTER_PORT \
tools/train.py configs/lst/ntu60_xsub_3dkp/j_vanilla.py --launcher pytorch --validate --seed 42


#torchrun --nproc_per_node=4 tools/train.py configs/mst/ntu60_xsub_3dkp/j1.py --launcher pytorch --validate --seed 42
#
#torchrun --nproc_per_node=4 tools/train.py configs/mst/ntu60_xsub_3dkp/j2.py --launcher pytorch --validate --seed 42
#
#torchrun --nproc_per_node=4 tools/train.py configs/mst/ntu60_xsub_3dkp/j3.py --launcher pytorch --validate --seed 42
#
#torchrun --nproc_per_node=4 tools/train.py configs/mst/ntu60_xsub_3dkp/j4.py --launcher pytorch --validate --seed 42

