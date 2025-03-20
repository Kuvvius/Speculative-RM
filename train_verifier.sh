#!/bin/bash


NNODES=1
GPUS_PER_NODE=1

echo "START TIME: $(date)"
TIMESTAMP="$(date "+%m-%d_%H-%M-%S")"

source activate ../miniconda3/envs/py310-pt201
python ./verifier_training_gsm8k.py \
    --max_epochs 5 \
    --gpus $GPUS_PER_NODE \
    --log_every_n_steps 10 \
    --precision 16 \
    --save_dir ./verifier_outputs_test \
    --save_top_k 3 \
    --monitor avg_train_loss \
    --mode min \
    --timestamp $TIMESTAMP \
    --gradient_clip_val 1.0 \
    --train \
    --num_nodes $NNODES \
    --accumulate_grad_batches 2 \
    --strategy ddp \
    --seed 19990303 \
    --verifier_loss LineMSE \
    --membank 100 \
    --model_type deberta \
    --model_name ./microsoft-deberta-v3-large \
    --lr 1e-5 \
    --l2 0. \
    --warmup 0.1 \
    --show_training_ex 100 \
    --scheduler linear \
    --data_dir ./data/prm800 \
    --num_workers 32 \
    --train_data ./prm800/train_2cls.jsonl \
    --micro_batch_size 8 \
    --task verifier \
    --recreate_dataset \
    

echo "END TIME: $(date)"
