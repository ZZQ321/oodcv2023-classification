#!/bin/sh		

dataset='robin'
port=3000
GPUS=1
lr='5e-6'
cfg='configs/deit_large.yaml'
root='deit_large'

source='train'
target='phase2-cls'
log_path="loghaha/${dataset}/${root}/${source}_${target}/${lr}"
out_path="resultshaha/${dataset}/${root}/${source}_${target}/${lr}"

python -m torch.distributed.run --nproc_per_node ${GPUS} --master_port ${port} dist_pmTrans.py --use-checkpoint \
--source ${source} --target ${target} --dataset ${dataset} --tag class32 --local_rank 0 --batch-size 16 --head_lr_ratio 10 --log-dir ${log_path} --output ${out_path} --cfg ${cfg}

