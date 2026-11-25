#!/usr/bin/env bash
# Pack the debate transcripts and GPU measurements of Phases 2 to 4 for the release.
# Works on a copy: the original result files are never changed.
#
# Usage (from anywhere):
#   bash tools/pack_release.sh OUT_DIR [options for tools/sanitize_release.py]
# Example:
#   bash tools/pack_release.sh ~/xmad_release --replace /cluster/home/user /path/to/workspace \
#       --node-prefix node --partition gpu_partition
set -euo pipefail

OUT="${1:?usage: bash tools/pack_release.sh OUT_DIR [sanitize options]}"
shift
REPO="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$OUT/staging"

mkdir -p "$OUT"
rm -rf "$STAGE"
rm -f "$OUT"/xmad_*.tar.gz "$OUT"/SHA256SUMS.txt
mkdir -p "$STAGE"

cd "$REPO"
echo "copying result files ..."
cp -r --parents 2_iteration/logs/debate_* 2_iteration/logs/divergence_comparison.json "$STAGE/"
cp -r --parents 3_iteration/results/*/sessions 3_iteration/results/*/gpu_metrics_*.csv "$STAGE/"
cp -r --parents 4_iteration/results/*/sessions 4_iteration/results/*/gpu_metrics_*.csv "$STAGE/"
rm -rf "$STAGE"/3_iteration/results/*_test "$STAGE"/4_iteration/results/*_test

echo "removing infrastructure details from the copy ..."
python3 "$REPO/tools/sanitize_release.py" "$STAGE" "$@"

echo "packing ..."
cd "$STAGE"
tar -czf "$OUT/xmad_phase2_transcripts.tar.gz" 2_iteration
tar -czf "$OUT/xmad_phase3_transcripts_gpu.tar.gz" 3_iteration
tar -czf "$OUT/xmad_phase4_transcripts_gpu.tar.gz" 4_iteration

cd "$OUT"
sha256sum xmad_*.tar.gz > SHA256SUMS.txt
rm -rf "$STAGE"
ls -lh "$OUT"
echo "done"
