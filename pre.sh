#!/bin/sh		

dataset='robin'
port=3008
GPUS=1
lr='5e-6'
cfg='configs/vit_large.yaml'
root='vit_large'

source='train'
target='phase2-cls'
log_path="loghaha/${dataset}/${root}/${source}_${target}/${lr}"
out_path="resultshaha/${dataset}/${root}/${source}_${target}/${lr}"

python -m torch.distributed.run --nproc_per_node ${GPUS} --master_port ${port} prediction.py --use-checkpoint \
--resume model_path \
--source ${source} --target ${target} --dataset ${dataset} --tag PM --local_rank 0 --batch-size 1 --head_lr_ratio 10 --log-dir ${log_path} --output ${out_path} --cfg ${cfg} --results_name large_0_32_2
