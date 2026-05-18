#!/bin/bash
# ─────────────────────────────────────────────
# FULL RUN: CombiGCN + Image Only similarity
# Data: CLIP embeddings (clip_10k_sample)
# Tương đương: hr/LightGCN_only_img.py
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
    --sim_type img_only \
    --wandb_run_name img_only_layers4_dim512_lr0.001_reg1e-04_clip \
    --hf_repo_id "$HF_REPO_ID" \
    --embed_size 512 \
    --layer_size "[512,512,512,512]" \
    --lr 0.001 \
    --regs "[1e-4]" \
    --batch_size 8192 \
    --epoch 1000 \
    --eval_interval 40 \
    --early_stop_steps 0 \
    --Ks "[1,5,10,20,50]" \
    --verbose 1 \
    --save_flag 1 \
    --checkpoint_interval 200 \
    --weights_path weights/ \
    --output_path output/ \
    --gpu_id 0
