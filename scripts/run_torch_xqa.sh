#!/bin/bash

OUTPUT_DIR="runs/xqa/afriteva_v2_large_squad_v2"

NUM_EPOCHS=5
LR=3e-5
BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=2
MAX_SEQ_LENGTH=512

EVAL_STRATEGY="no"
EVAL_STEPS=1000
SAVE_STRATEGY="epoch"
LOGGING_STEPS=1000

mkdir -p "$OUTPUT_DIR"

python -m teva.torch_module.xqa \
    --model_name_or_path castorini/afriteva_v2_large \
    --output_dir "${OUTPUT_DIR}" \
    --dataset_name "squad_v2" \
    --do_train \
    --learning_rate ${LR} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --num_train_epochs ${NUM_EPOCHS} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --predict_with_generate True \
    --question_column question \
    --answer_column answers \
    --context_column context \
    --doc_stride 128 \
    --logging_steps ${LOGGING_STEPS} \
    --eval_strategy "${EVAL_STRATEGY}" \
    --eval_steps ${EVAL_STEPS} \
    --save_strategy "${SAVE_STRATEGY}" \
    --version_2_with_negative \
    --report_to "none"
