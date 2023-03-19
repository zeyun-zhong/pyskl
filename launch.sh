#!/usr/bin/env bash

torchrun --nproc_per_node=4 tools/train.py configs/mst/ntu60_xsub_3dkp/j.py --launcher pytorch --validate --seed 42

torchrun --nproc_per_node=4 tools/train.py configs/mst/ntu60_xsub_3dkp/j1.py --launcher pytorch --validate --seed 42

torchrun --nproc_per_node=4 tools/train.py configs/mst/ntu60_xsub_3dkp/j2.py --launcher pytorch --validate --seed 42

torchrun --nproc_per_node=4 tools/train.py configs/mst/ntu60_xsub_3dkp/j3.py --launcher pytorch --validate --seed 42

torchrun --nproc_per_node=4 tools/train.py configs/mst/ntu60_xsub_3dkp/j4.py --launcher pytorch --validate --seed 42

