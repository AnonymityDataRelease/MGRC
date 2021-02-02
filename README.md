# MSRC -- Multi-stage Reading Comprehension
This repo is for our paper MSRC:A Multi-Stage Reading Comprehension Model for Question Answering



# Train and Eval

Codes are based on huggingface framerwork.



 python run_msrc.py \
    --model_type bert \
    --model_name_or_path bert-large-uncased \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file data/train-v2.0.json \
    --predict_file data/dev-v2.0.json \
    --learning_rate 3e-5 \
    --num_train_epochs 5 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./models/output \
    --per_gpu_eval_batch_size=4   \
    --per_gpu_train_batch_size=4  \
    --gradient_accumulation_steps=1 \
    --version_2_with_negative \
    --save_steps=1000 \
    --overwrite_output_dir \
    --eval_all_checkpoints

# Files
1. utils_q_type.py: used to label the question type.
2. run_msrc.py and utils_msrc.py: used train and eval MSRC model.
3. mymodel.py: the designed MSRC model.

