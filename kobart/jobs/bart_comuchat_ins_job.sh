#!/bin/bash

echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# conda 환경 활성화.
source ~/.bashrc
conda activate grad_nlp_comu_bart

# cuda 11.0 환경 구성.
ml purge
ml load cuda/11.0

# 활성화된 환경에서 코드 실행.
python kobart_chit_chat.py --gradient_clip_val 1.0 --max_epochs 20 --gpus 1 --task_prefix comuchat_ins

echo "###"
echo "### END DATE=$(date)"