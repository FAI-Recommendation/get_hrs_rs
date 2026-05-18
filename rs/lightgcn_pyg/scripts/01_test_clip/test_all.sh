#!/bin/bash
# ─────────────────────────────────────────────
# TEST ALL: Chạy nhanh cả 4 variants (20 epochs mỗi cái)
# Data: CLIP embeddings (clip_10k_sample)
# Dùng để verify toàn bộ pipeline hoạt động trước khi run_all_clip
# ─────────────────────────────────────────────

set -e

[ -d /opt/conda/bin ] && export PATH="/opt/conda/bin:$PATH"

SCRIPT_DIR="$(dirname "$0")"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 TEST 1/4: LightGCN thuần (--sim_type none)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bash "$SCRIPT_DIR/test_lightgcn.sh"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 TEST 2/4: CombiGCN + img_only"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bash "$SCRIPT_DIR/test_img_only.sh"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 TEST 3/4: CombiGCN + multimodal"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bash "$SCRIPT_DIR/test_multimodal.sh"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 TEST 4/4: CombiGCN + tfidf"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bash "$SCRIPT_DIR/test_tfidf.sh"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ ALL 4 TESTS PASSED!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
