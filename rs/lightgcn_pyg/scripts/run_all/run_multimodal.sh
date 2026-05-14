#!/bin/bash
# ─────────────────────────────────────────────
# FULL RUN: CombiGCN + Multimodal (BERT text + Image)
# Tương đương: hr/LightGCN_bert_img.py
# ─────────────────────────────────────────────

cd "$(dirname "$0")/../.."

# Load environment variables from repo root .env if present
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

python train.py \
    --data_path ../../get10k_data/clip_10k_sample \
    --dataset "" \
    --sim_type multimodal \
    --embed_size 64 \
    --layer_size "[64,64,64]" \
    --lr 0.001 \
    --regs "[1e-5]" \
    --batch_size 1024 \
    --epoch 1000 \
    --eval_interval 10 \
    --early_stop_steps 5 \
    --Ks "[1,5,10,20,50]" \
    --verbose 1 \
    --save_flag 1 \
    --weights_path weights/ \
    --output_path output/ \
    --gpu_id 0
