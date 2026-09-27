#!/usr/bin/env bash
set -eu
cd /home/yash/dev/clg/amlc-2026
TEAM="${1:-TEAM_NAME}"
PKG="/tmp/${TEAM}_submission"
rm -rf "$PKG"; mkdir -p "$PKG/output" "$PKG/code/business_entity_resolution/src"

cp output/matching_results.tsv output/candidate_pairs.tsv "$PKG/output/"
cp -r blocking matcher "$PKG/code/business_entity_resolution/src/"
cp README.md requirements.txt "$PKG/code/business_entity_resolution/"
cp run_submission.sh finish_india.sh "$PKG/code/business_entity_resolution/" 2>/dev/null || true
cp Documentation_template.md "$PKG/"

# strip anything that is not source: caches, data, models, logs, intermediates
find "$PKG/code" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
rm -rf "$PKG/code/business_entity_resolution/src/matcher/data" \
       "$PKG/code/business_entity_resolution/src/matcher/results" \
       "$PKG/code/business_entity_resolution/src/blocking/results"
# keep the trained model -- it is needed to reproduce without retraining
mkdir -p "$PKG/code/business_entity_resolution/src/matcher/models"
cp matcher/models/india_kb150.lgb matcher/models/india_kb150_importance.json \
   "$PKG/code/business_entity_resolution/src/matcher/models/" 2>/dev/null || true
cp matcher/data/india_kb150_cols.json matcher/data/india_kb150_idf.pkl \
   "$PKG/code/business_entity_resolution/src/matcher/models/" 2>/dev/null || true

cd /tmp && rm -f "${TEAM}_submission.zip"
zip -qr "${TEAM}_submission.zip" "${TEAM}_submission"
echo "=== PACKAGE BUILT ==="
ls -la "/tmp/${TEAM}_submission.zip" | awk '{printf "%s  %.1f MB\n",$9,$5/1048576}'
echo "--- contents ---"
cd "$PKG" && find . -type f | sed 's|^\./||' | sort | head -40
echo "--- file counts ---"
echo "src files: $(find code -type f | wc -l)"
