"""Generate BM3 and FREEDOM scripts (run + test) for all sim_type x embed_type combos."""
import os

BASE = "rs/lightgcn_pyg/scripts"

SIM_TYPES = ["img_only", "tfidf", "multimodal", "multimodal_attention"]
ENCODERS = [
    ("clip",  "../get10k_data/clip_10k_sample"),
    ("mbnv2", "/root/get_hrs_rs/rs/get10k_data/output_10k_sample"),
]
MODELS = [
    ("bm3",     3),   # folder prefix
    ("freedom", 5),
]

ENV_BLOCK = """\
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
"""

ENV_BLOCK_TEST = """\
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi
"""

PAPER = {
    "bm3":     "Bootstrap Latent Representations for Multi-modal Recommendation, WWW 2023",
    "freedom": "FREEDOM: Freezing and Denoising Graph Structures for Multimodal Recommendation, ACM MM 2023",
}

BM3_EXTRA_RUN  = "    --bm3_momentum 0.995 \\\n    --bm3_cl_weight 0.2 \\"
BM3_EXTRA_TEST = "    --bm3_momentum 0.995 \\\n    --bm3_cl_weight 0.2 \\"
FREE_EXTRA_RUN  = "    --freedom_knn_k 10 \\\n    --freedom_cl_weight 0.1 \\\n    --freedom_cl_temp 0.2 \\"
FREE_EXTRA_TEST = "    --freedom_knn_k 10 \\\n    --freedom_cl_weight 0.1 \\\n    --freedom_cl_temp 0.2 \\"

def make_run(model, embed_type, data_path, sim_type):
    extra = BM3_EXTRA_RUN if model == "bm3" else FREE_EXTRA_RUN
    run_name = f"{model}_{sim_type}_layers4_dim512_lr0.001_reg1e-04_{embed_type}"
    return f"""\
#!/bin/bash
# ─────────────────────────────────────────────
# FULL RUN: {model.upper()} + {sim_type}
# Data: {embed_type} embeddings
# Paper: {PAPER[model]}
# ─────────────────────────────────────────────

cd "$(dirname "$0")/../.."

[ -d /opt/conda/bin ] && export PATH="/opt/conda/bin:$PATH"

{ENV_BLOCK}
python3 train.py \\
    --model {model} \\
    --embed_type {embed_type} \\
    --sim_type {sim_type} \\
    --data_path {data_path} \\
    --dataset "" \\
    --wandb_run_name {run_name} \\
    --hf_repo_id "$HF_REPO_ID" \\
    --embed_size 512 \\
    --layer_size "[512,512,512,512]" \\
    --lr 0.001 \\
    --regs "[1e-4]" \\
    --batch_size 8192 \\
    --epoch 1000 \\
    --eval_interval 40 \\
    --early_stop_steps 0 \\
    --Ks "[1,5,10,20]" \\
{extra}
    --verbose 1 \\
    --save_flag 1 \\
    --checkpoint_interval 200 \\
    --weights_path weights/ \\
    --output_path output/ \\
    --gpu_id 0
"""

def make_test(model, embed_type, data_path, sim_type):
    extra = BM3_EXTRA_TEST if model == "bm3" else FREE_EXTRA_TEST
    return f"""\
#!/bin/bash
# ─────────────────────────────────────────────
# TEST: {model.upper()} + {sim_type} ({embed_type}) — 20 epochs để verify pipeline
# ─────────────────────────────────────────────

cd "$(dirname "$0")/../.."

[ -d /opt/conda/bin ] && export PATH="/opt/conda/bin:$PATH"

{ENV_BLOCK_TEST}
python3 train.py \\
    --model {model} \\
    --embed_type {embed_type} \\
    --sim_type {sim_type} \\
    --data_path {data_path} \\
    --dataset "" \\
    --embed_size 64 \\
    --layer_size "[64,64,64]" \\
    --lr 0.001 \\
    --regs "[1e-4]" \\
    --batch_size 1024 \\
    --epoch 20 \\
    --eval_interval 10 \\
    --early_stop_steps 3 \\
    --Ks "[1,5,10,20]" \\
{extra}
    --verbose 1 \\
    --save_flag 0 \\
    --gpu_id 0
"""

def make_run_all(model, embed_type, folder):
    lines = []
    for i, sim in enumerate(SIM_TYPES, 1):
        lines.append(f'echo ""\necho "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"\necho "🏃 {i}/{len(SIM_TYPES)}: {model.upper()} + {sim}"\necho "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"\nbash "$SCRIPT_DIR/run_{sim}.sh"')
    results = "\n".join([f'   output/{model}_{sim}_{embed_type}/{model}.result' for sim in SIM_TYPES])
    body = "\n\n".join(lines)
    return f"""\
#!/bin/bash
# ─────────────────────────────────────────────
# RUN ALL: {model.upper()} — 4 sim_type variants tuần tự
# Data: {embed_type} embeddings
# ─────────────────────────────────────────────

set -e

[ -d /opt/conda/bin ] && export PATH="/opt/conda/bin:$PATH"

SCRIPT_DIR="$(dirname "$0")"
START_TIME=$(date +%s)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 FULL TRAINING — {model.upper()} x 4 sim_type ({embed_type}, 1000 epochs)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

{body}

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ ALL DONE!  ($((ELAPSED / 60))m $((ELAPSED % 60))s)"
echo ""
echo "📄 Results:"
{chr(10).join([f'echo "   output/{model}_{sim}_{embed_type}/{model}.result"' for sim in SIM_TYPES])}
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
"""

def make_test_all(model, embed_type):
    lines = []
    for i, sim in enumerate(SIM_TYPES, 1):
        lines.append(f'echo ""\necho "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"\necho "🧪 TEST {i}/{len(SIM_TYPES)}: {model.upper()} + {sim}"\necho "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"\nbash "$SCRIPT_DIR/test_{sim}.sh"')
    body = "\n\n".join(lines)
    return f"""\
#!/bin/bash
# ─────────────────────────────────────────────
# TEST ALL: {model.upper()} — 4 sim_type variants (20 epochs mỗi cái)
# ─────────────────────────────────────────────

set -e

[ -d /opt/conda/bin ] && export PATH="/opt/conda/bin:$PATH"

SCRIPT_DIR="$(dirname "$0")"

{body}

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ ALL {len(SIM_TYPES)} TESTS PASSED!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
"""

# ── Generate all files ──
encoder_idx = {"clip": 0, "mbnv2": 1}

for model, prefix in MODELS:
    for enc_idx, (embed_type, data_path) in enumerate(ENCODERS):
        run_folder_num = prefix + enc_idx
        test_folder_num = prefix + enc_idx

        run_folder  = f"{BASE}/{run_folder_num:02d}_run_{model}_{embed_type}"
        test_folder = f"{BASE}/{test_folder_num:02d}_test_{model}_{embed_type}"

        os.makedirs(run_folder,  exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)

        # Individual scripts
        for sim_type in SIM_TYPES:
            run_path  = f"{run_folder}/run_{sim_type}.sh"
            test_path = f"{test_folder}/test_{sim_type}.sh"

            with open(run_path,  "w", encoding="utf-8", newline="\n") as f:
                f.write(make_run(model, embed_type, data_path, sim_type))

            with open(test_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(make_test(model, embed_type, data_path, sim_type))

        # run_all.sh
        with open(f"{run_folder}/run_all.sh",   "w", encoding="utf-8", newline="\n") as f:
            f.write(make_run_all(model, embed_type, run_folder))

        # test_all.sh
        with open(f"{test_folder}/test_all.sh", "w", encoding="utf-8", newline="\n") as f:
            f.write(make_test_all(model, embed_type))

        print(f"Generated: {run_folder}  (4 run + run_all)")
        print(f"Generated: {test_folder} (4 test + test_all)")

print("Done!")
