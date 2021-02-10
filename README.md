
# MGRC

Code for the paper "MGRC: An End-to-End Multi-Granularity Reading Comprehension Model for Question Answering" (sumbitted to TNNLS)

**Note:** Our code is based on the [huggingface](https://github.com/huggingface/transformers). More examples are available [here](https://github.com/huggingface/transformers/tree/master/examples/question-answering). 

This example code fine-tunes BERT on the SQuAD2.0 dataset.
```
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
    --per_gpu_eval_batch_size=32   \
    --per_gpu_train_batch_size=32  \
    --gradient_accumulation_steps=1 \
    --version_2_with_negative \
    --save_steps=1000 \
    --overwrite_output_dir \
    --eval_all_checkpoints
```
**Description of files:**

 1. utils_q_type.py: codes used to label the question type.
 2. run_msrc.py and utils_msrc.py: used train and eval MGRC model.
 3. mymodel.py: codes of the designed MGRC model.
 4. myvocab.txt: the vocabulary for MGRC model. Compared with the original vocabulary of BERT, it adds the designed special tokens, such as [QPAR], [QSEN], [QANS] etc.

