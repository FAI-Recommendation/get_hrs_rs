#!/bin/bash
# ─────────────────────────────────────────────
# FULL RUN: CombiGCN + Multimodal (BERT text + MobileNetV2 image)
# Data: MobileNetV2 embeddings (output_10k_sample)
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
    --data_path /root/get_hrs_rs/rs/get10k_data/output_10k_sample \
    --dataset "" \
    --sim_type multimodal \
    --wandb_run_name multimodal_layers4_dim512_lr0.001_reg1e-05_mbnv2 \
    --hf_repo_id "$HF_REPO_ID_MBNV2" \
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
    --checkpoint_interval 200 \
    --weights_path weights/ \
    --output_path output/ \
    --gpu_id 0
