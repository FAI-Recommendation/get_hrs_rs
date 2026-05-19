#!/bin/bash
# ─────────────────────────────────────────────
# RUN ALL: FREEDOM — 4 sim_type variants tuần tự
# Data: clip embeddings
# ─────────────────────────────────────────────

set -e

[ -d /opt/conda/bin ] && export PATH="/opt/conda/bin:$PATH"

SCRIPT_DIR="$(dirname "$0")"
START_TIME=$(date +%s)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 FULL TRAINING — FREEDOM x 4 sim_type (clip, 1000 epochs)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏃 1/4: FREEDOM + img_only"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bash "$SCRIPT_DIR/run_img_only.sh"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏃 2/4: FREEDOM + tfidf"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bash "$SCRIPT_DIR/run_tfidf.sh"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏃 3/4: FREEDOM + multimodal"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bash "$SCRIPT_DIR/run_multimodal.sh"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏃 4/4: FREEDOM + multimodal_attention"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bash "$SCRIPT_DIR/run_multimodal_attention.sh"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ ALL DONE!  ($((ELAPSED / 60))m $((ELAPSED % 60))s)"
echo ""
echo "📄 Results:"
echo "   output/freedom_img_only_clip/freedom.result"
echo "   output/freedom_tfidf_clip/freedom.result"
echo "   output/freedom_multimodal_clip/freedom.result"
echo "   output/freedom_multimodal_attention_clip/freedom.result"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
