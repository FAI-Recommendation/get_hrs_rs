#!/bin/bash
# ─────────────────────────────────────────────
# FULL RUN: CombiGCN + Multimodal (BERT text + Image)
# Tương đương: hr/LightGCN_bert_img.py
# ─────────────────────────────────────────────

cd "$(dirname "$0")/../.."

[ -d /opt/conda/bin ] && export PATH="/opt/conda/bin:$PATH"

# Load .env: thu muc hien tai (rs/lightgcn_pyg/) hoac repo root (../../)
for _env in ".env" "../../.env"; do
    if [ -f "$_env" ]; then
        set -a
        # shellcheck disable=SC1091
        source "$_env"
        set +a
        echo "   📋 Loaded env: $_env"
        break
    fi
done

python3 train.py \
    --data_path ../get10k_data/clip_10k_sample \
    --dataset "" \
    --sim_type multimodal \
    --embed_size 512 \
    --layer_size "[512,512,512,512]" \
    --lr 0.001 \
    --regs "[1e-5]" \
    --batch_size 8192 \
    --epoch 500 \
    --eval_interval 20 \
    --early_stop_steps 0 \
    --Ks "[1,5,10,20,50]" \
    --verbose 1 \
    --save_flag 1 \
    --checkpoint_interval 100 \
    --weights_path weights/ \
    --output_path output/ \
    --gpu_id 0
