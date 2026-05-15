#!/bin/bash
# ─────────────────────────────────────────────
# TEST: CombiGCN + Image Only similarity
# Chạy nhanh 20 epochs để kiểm tra pipeline
# ─────────────────────────────────────────────

cd "$(dirname "$0")/../.."

[ -d /opt/conda/bin ] && export PATH="/opt/conda/bin:$PATH"

# Load environment variables from repo root .env if present
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

python3 train.py \
    --data_path ../get10k_data/clip_10k_sample \
    --dataset "" \
    --sim_type img_only \
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
