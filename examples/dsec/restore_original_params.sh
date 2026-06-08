#!/usr/bin/env bash
# Restore cuVSLAM SOF parameters to their original cuVSLAM 15.0 defaults.
# Run this after the sweep to get back to the unmodified state.
# To apply the best sweep result instead, run apply_best_params.sh.

set -euo pipefail

REPO=/home/loay/cuVSLAM
SOF_GPU_CPP=${REPO}/libs/sof/sof_mono_gpu.cpp
SOF_CFG_H=${REPO}/libs/sof/sof_config.h
SOF_CMAKE=${REPO}/libs/sof/CMakeLists.txt
BUILD_DIR=${REPO}/build

echo "[restore] NCC thresholds → 0.80 / 0.85"
sed -i \
    "s/static constexpr float NCC_PRIMARY_THRESHOLD = [0-9.]*f;/static constexpr float NCC_PRIMARY_THRESHOLD = 0.80f;/" \
    "$SOF_GPU_CPP"
sed -i \
    "s/static constexpr float NCC_RETRACK_THRESHOLD = [0-9.]*f;/static constexpr float NCC_RETRACK_THRESHOLD = 0.85f;/" \
    "$SOF_GPU_CPP"

echo "[restore] num_desired_tracks → 450"
sed -i \
    "s/int32_t num_desired_tracks = [0-9]*;/int32_t num_desired_tracks = 450;/" \
    "$SOF_CFG_H"

echo "[restore] removing SOF_USE_SMALLER_NCC from CMakeLists"
sed -i '/SOF_USE_SMALLER_NCC/d' "$SOF_CMAKE"
# Also remove the comment line above it
sed -i '/Smaller 3x3 NCC window/d' "$SOF_CMAKE"

echo "[build] rebuilding..."
make -C "$BUILD_DIR" -j"$(nproc)" 2>&1 | tail -4
CUVSLAM_BUILD_DIR="$BUILD_DIR" pip install -e "${REPO}/python/" -q

echo "Done. Parameters restored to original cuVSLAM 15.0 defaults."
