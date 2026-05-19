#!/bin/bash
# ─────────────────────────────────────────────
# FULL RUN: FREEDOM + img_only
# Data: clip embeddings
# Paper: FREEDOM: Freezing and Denoising Graph Structures for Multimodal Recommendation, ACM MM 2023
# ─────────────────────────────────────────────

cd "$(dirname "$0")/../.."

[ -d /opt/conda/bin ] && export PATH="/opt/conda/bin:$PATH"

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
    --model freedom \
    --embed_type clip \
    --sim_type img_only \
    --data_path ../get10k_data/clip_10k_sample \
    --dataset "" \
    --wandb_run_name freedom_img_only_layers4_dim512_lr0.001_reg1e-04_clip \
    --hf_repo_id "$HF_REPO_ID" \
    --embed_size 512 \
    --layer_size "[512,512,512,512]" \
    --lr 0.001 \
    --regs "[1e-4]" \
    --batch_size 8192 \
    --epoch 1000 \
    --eval_interval 40 \
    --early_stop_steps 0 \
    --Ks "[1,5,10,20]" \
    --freedom_knn_k 10 \
    --freedom_cl_weight 0.1 \
    --freedom_cl_temp 0.2 \
    --verbose 1 \
    --save_flag 1 \
    --checkpoint_interval 200 \
    --weights_path weights/ \
    --output_path output/ \
    --gpu_id 0
