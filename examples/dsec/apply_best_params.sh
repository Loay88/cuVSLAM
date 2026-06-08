#!/usr/bin/env bash
# Apply the best sweep parameters (Run 6: NCC 0.55/0.60, num_tracks=200, RMSE=45.98m).
# Edit the values below to apply a different combination.

set -euo pipefail

REPO=/home/loay/cuVSLAM
SOF_GPU_CPP=${REPO}/libs/sof/sof_mono_gpu.cpp
SOF_CFG_H=${REPO}/libs/sof/sof_config.h
BUILD_DIR=${REPO}/build

NCC_PRIMARY=0.55
NCC_RETRACK=0.60
NUM_TRACKS=200

echo "[apply] NCC thresholds → ${NCC_PRIMARY} / ${NCC_RETRACK}"
sed -i \
    "s/static constexpr float NCC_PRIMARY_THRESHOLD = [0-9.]*f;/static constexpr float NCC_PRIMARY_THRESHOLD = ${NCC_PRIMARY}f;/" \
    "$SOF_GPU_CPP"
sed -i \
    "s/static constexpr float NCC_RETRACK_THRESHOLD = [0-9.]*f;/static constexpr float NCC_RETRACK_THRESHOLD = ${NCC_RETRACK}f;/" \
    "$SOF_GPU_CPP"

echo "[apply] num_desired_tracks → ${NUM_TRACKS}"
sed -i \
    "s/int32_t num_desired_tracks = [0-9]*;/int32_t num_desired_tracks = ${NUM_TRACKS};/" \
    "$SOF_CFG_H"

echo "[build] rebuilding..."
make -C "$BUILD_DIR" -j"$(nproc)" 2>&1 | tail -4
CUVSLAM_BUILD_DIR="$BUILD_DIR" pip install -e "${REPO}/python/" -q

echo "Done. Best sweep parameters applied (RMSE ~45.98m on zurich_city_09_d events)."
