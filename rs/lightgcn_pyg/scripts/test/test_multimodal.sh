#!/bin/bash
# ─────────────────────────────────────────────
# TEST: CombiGCN + Multimodal (BERT text + Image)
# Chạy nhanh 20 epochs để kiểm tra pipeline
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
    --epoch 20 \
    --eval_interval 10 \
    --early_stop_steps 3 \
    --Ks "[1,5,10,20]" \
    --verbose 1 \
    --save_flag 0 \
    --gpu_id 0
