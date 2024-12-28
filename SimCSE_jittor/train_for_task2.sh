#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.
#    --metric_for_best_model stsb_spearman \

# Attention: trainers_jt.py line 202, 298, 302 has been modified 
# Atteintion: models_bert.py 242

# 请选择Task2/bert为多语言bert

python train_jt_task2.py \
    --model_name_or_path Task2/bert \ 
    --train_file Task2/es_large.txt \
    --output_dir Task2/result/es_large \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    "$@"
