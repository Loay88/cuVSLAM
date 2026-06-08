#!/usr/bin/env bash
# Parameter sweep for event time-surface tracking in cuVSLAM.
# Varies NCC threshold and num_desired_tracks; rebuilds and evaluates each combination.
#
# Usage: bash sweep_event_params.sh
# Results written to sweep_results.csv

set -euo pipefail

REPO=/home/loay/cuVSLAM
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SEQ=/home/loay/research/DSEC/dataset/zurich_city_09_d
IMAGE_DIR=event_timesurface_30_calib_backward
TIMESTAMPS=event_timesurface_30_calib_backward/timestamps.csv
CALIB=zurich_city_09_d_calibration/cam_to_cam.yaml
CAMERA_KEY=camRect3
GT=/home/loay/research/DSEC/dataset/zurich_city_09_d/results/lidar_imu_09_d_poses_tum_camRect3.txt

SOF_GPU_CPP=${REPO}/libs/sof/sof_mono_gpu.cpp
SOF_CFG_H=${REPO}/libs/sof/sof_config.h
BUILD_DIR=${REPO}/build
OUTPUT_CSV=${SCRIPT_DIR}/sweep_results.csv

NCC_PRIMARY_VALUES=(0.50 0.55 0.60 0.65)
NUM_TRACKS_VALUES=(100 150 200)

# --------------------------------------------------------------------------
echo "run_id,ncc_primary,ncc_retrack,num_tracks,rmse_m,mean_m,median_m" > "$OUTPUT_CSV"

run_id=0
for ncc_p in "${NCC_PRIMARY_VALUES[@]}"; do
    # keep retrack 0.05 above primary, capped at 0.80
    ncc_r=$(python3 -c "print(f'{min($ncc_p + 0.05, 0.80):.2f}')")

    for n_tracks in "${NUM_TRACKS_VALUES[@]}"; do
        run_id=$((run_id + 1))
        echo ""
        echo "============================================================"
        echo " Run ${run_id}: ncc_primary=${ncc_p}  ncc_retrack=${ncc_r}  num_tracks=${n_tracks}"
        echo "============================================================"

        # -- 1. Patch sof_mono_gpu.cpp NCC constants --
        sed -i \
            "s/static constexpr float NCC_PRIMARY_THRESHOLD = [0-9.]*f;/static constexpr float NCC_PRIMARY_THRESHOLD = ${ncc_p}f;/" \
            "$SOF_GPU_CPP"
        sed -i \
            "s/static constexpr float NCC_RETRACK_THRESHOLD = [0-9.]*f;/static constexpr float NCC_RETRACK_THRESHOLD = ${ncc_r}f;/" \
            "$SOF_GPU_CPP"

        # -- 2. Patch sof_config.h num_desired_tracks --
        sed -i \
            "s/int32_t num_desired_tracks = [0-9]*;/int32_t num_desired_tracks = ${n_tracks};/" \
            "$SOF_CFG_H"

        # -- 3. Build (incremental) --
        echo "[build] make -j$(nproc)"
        make -C "$BUILD_DIR" -j"$(nproc)" 2>&1 | tail -5

        echo "[install] pip install"
        CUVSLAM_BUILD_DIR="$BUILD_DIR" pip install -e "${REPO}/python/" -q

        # -- 4. Run tracker --
        OUT_TUM=/tmp/sweep_run_${run_id}.tum
        python3 "${SCRIPT_DIR}/track_dsec_mono.py" \
            --sequence "$SEQ" \
            --image-dir "$IMAGE_DIR" \
            --timestamps "$TIMESTAMPS" \
            --calibration "$CALIB" \
            --camera-key "$CAMERA_KEY" \
            --no-viz \
            --no-motion-model \
            --output "$OUT_TUM"

        # -- 5. Evaluate --
        echo "[eval] evo_ape"
        METRICS=$(evo_ape tum "$GT" "$OUT_TUM" -as 2>&1 | grep -E "^\s+(rmse|mean|median)" || echo "rmse N/A mean N/A median N/A")
        RMSE=$(echo "$METRICS"   | grep rmse   | awk '{print $2}' || echo "N/A")
        MEAN=$(echo "$METRICS"   | grep mean   | awk '{print $2}' || echo "N/A")
        MEDIAN=$(echo "$METRICS" | grep median | awk '{print $2}' || echo "N/A")

        echo "  → rmse=${RMSE}  mean=${MEAN}  median=${MEDIAN}"
        echo "${run_id},${ncc_p},${ncc_r},${n_tracks},${RMSE},${MEAN},${MEDIAN}" >> "$OUTPUT_CSV"
    done
done

echo ""
echo "============================================================"
echo " Sweep complete. Results:"
echo "============================================================"
column -t -s, "$OUTPUT_CSV"

# Print best result
BEST=$(tail -n +2 "$OUTPUT_CSV" | sort -t, -k5 -n | head -1)
echo ""
echo "Best: $BEST"
