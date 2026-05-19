#!/bin/bash
# ─────────────────────────────────────────────
# TEST: FREEDOM với mbnv2 — 20 epochs để verify pipeline
# ─────────────────────────────────────────────

cd "$(dirname "$0")/../.."

[ -d /opt/conda/bin ] && export PATH="/opt/conda/bin:$PATH"

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

python3 train.py \
    --model freedom \
    --embed_type mbnv2 \
    --data_path ../get10k_data/mbnv2_10k_sample \
    --dataset "" \
    --embed_size 64 \
    --layer_size "[64,64,64]" \
    --lr 0.001 \
    --regs "[1e-4]" \
    --batch_size 1024 \
    --epoch 20 \
    --eval_interval 10 \
    --early_stop_steps 3 \
    --Ks "[1,5,10,20]" \
    --freedom_knn_k 10 \
    --freedom_cl_weight 0.1 \
    --freedom_cl_temp 0.2 \
    --verbose 1 \
    --save_flag 0 \
    --gpu_id 0
