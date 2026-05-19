#!/bin/bash
# ─────────────────────────────────────────────
# RUN ALL: BM3 — 4 sim_type variants tuần tự
# Data: clip embeddings
# ─────────────────────────────────────────────

set -e

[ -d /opt/conda/bin ] && export PATH="/opt/conda/bin:$PATH"

SCRIPT_DIR="$(dirname "$0")"
START_TIME=$(date +%s)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 FULL TRAINING — BM3 x 4 sim_type (clip, 1000 epochs)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏃 1/4: BM3 + img_only"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bash "$SCRIPT_DIR/run_img_only.sh"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏃 2/4: BM3 + tfidf"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bash "$SCRIPT_DIR/run_tfidf.sh"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏃 3/4: BM3 + multimodal"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bash "$SCRIPT_DIR/run_multimodal.sh"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏃 4/4: BM3 + multimodal_attention"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bash "$SCRIPT_DIR/run_multimodal_attention.sh"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ ALL DONE!  ($((ELAPSED / 60))m $((ELAPSED % 60))s)"
echo ""
echo "📄 Results:"
echo "   output/bm3_img_only_clip/bm3.result"
echo "   output/bm3_tfidf_clip/bm3.result"
echo "   output/bm3_multimodal_clip/bm3.result"
echo "   output/bm3_multimodal_attention_clip/bm3.result"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
