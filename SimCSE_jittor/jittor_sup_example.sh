#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.

python train_jt.py \
    --model_name_or_path bert \
    --train_file data/nli_for_simcse.csv \
    --output_dir result/my-sup-simcse-bert-base-uncased-jittor-tb \
    --num_train_epochs 3 \
    --per_device_train_batch_size 64 \
    --learning_rate 5e-5 \
    --max_seq_length 32 \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    "$@"
