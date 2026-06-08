"""
Monocular visual odometry on a DSEC sequence using cuVSLAM.

Minimal required arguments:
    python track_dsec_mono.py --sequence /path/to/zurich_city_04_a

With explicit paths (if your layout differs from standard DSEC):
    python track_dsec_mono.py \\
        --sequence /data/dsec/zurich_city_04_a \\
        --image-dir images/left/rectified \\
        --timestamps images/timestamps.txt \\
        --calibration calibration/cam_to_cam.yaml \\
        --camera-key camRect0

Output:
    Trajectory in TUM format (timestamp tx ty tz qx qy qz qw), one pose per line.
    Timestamps are in seconds (float).

Visualization:
    Requires rerun-sdk. Omit --no-viz to open the viewer.
    Use --save-rrd to also write an .rrd file.
"""

import argparse
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

import cuvslam
from dsec_utils import (
    build_frame_sequence,
    default_calibration_path,
    default_image_dir,
    default_timestamps_path,
    load_camera,
    load_frame,
)

# ---------------------------------------------------------------------------
# Optional rerun visualization
# ---------------------------------------------------------------------------

try:
    import rerun as rr
    import rerun.blueprint as rrb
    HAS_RERUN = True
except ImportError:
    HAS_RERUN = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _color_from_id(identifier: int) -> List[int]:
    return [
        (identifier * 17) % 256,
        (identifier * 31) % 256,
        (identifier * 47) % 256,
    ]


def _intrinsics_matrix(camera: cuvslam.Camera) -> np.ndarray:
    fx, fy = camera.focal
    cx, cy = camera.principal
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# Visualization setup
# ---------------------------------------------------------------------------

def setup_visualizer(save_rrd: Optional[str]) -> None:
    rr.init("cuVSLAM DSEC Mono", strict=True, spawn=True)
    if save_rrd is not None:
        rr.save(save_rrd)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.TimePanel(state="collapsed"),
            rrb.Vertical(
                row_shares=[0.6, 0.4],
                contents=[
                    rrb.Spatial3DView(origin="world"),
                    rrb.Spatial2DView(origin="world/cam0"),
                ],
            ),
        )
    )


def log_frame(
    frame_id: int,
    pose,
    image: np.ndarray,
    observations,
    landmarks,
    final_landmarks,
    trajectory: List,
    camera: cuvslam.Camera,
) -> None:
    obs_uv = [[o.u, o.v] for o in observations]
    obs_colors = [_color_from_id(o.id) for o in observations]
    lm_xyz = [lm.coords for lm in landmarks]
    lm_colors = [_color_from_id(lm.id) for lm in landmarks]

    rr.set_time_sequence("frame", frame_id)
    rr.log("world/trajectory", rr.LineStrips3D(trajectory))
    rr.log(
        "world/rig",
        rr.Transform3D(translation=pose.translation, quaternion=pose.rotation),
    )
    rr.log(
        "world/final_landmarks",
        rr.Points3D(list(final_landmarks.values()), radii=0.1),
    )
    rr.log("world/landmarks", rr.Points3D(lm_xyz, radii=0.2, colors=lm_colors))
    rr.log(
        "world/cam0",
        rr.Pinhole(
            image_plane_distance=1.0,
            image_from_camera=_intrinsics_matrix(camera),
            width=camera.size[0],
            height=camera.size[1],
        ),
    )
    rr.log("world/cam0/image", rr.Image(image).compress(jpeg_quality=80))
    rr.log("world/cam0/observations", rr.Points2D(obs_uv, radii=5, colors=obs_colors))
    rr.log("metrics/feature_count", rr.Scalar(len(observations)))


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def save_tum(path: str, trajectory: List[List[float]]) -> None:
    """
    Save trajectory in TUM format.

    Each row: timestamp_s tx ty tz qx qy qz qw
    Timestamp is in seconds (float) as required by TUM evaluation tools.
    """
    rows = []
    for timestamp_ns, tx, ty, tz, qx, qy, qz, qw in trajectory:
        rows.append([timestamp_ns / 1e9, tx, ty, tz, qx, qy, qz, qw])
    np.savetxt(path, rows, fmt="%.9f")
    print(f"Saved TUM trajectory ({len(rows)} poses) → {path}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--sequence",
        required=True,
        help="Path to the DSEC sequence root directory.",
    )
    p.add_argument(
        "--image-dir",
        default=None,
        help=(
            "Relative path (inside --sequence) to rectified left images. "
            "Default: images/left/rectified"
        ),
    )
    p.add_argument(
        "--timestamps",
        default=None,
        help=(
            "Relative path (inside --sequence) to timestamps.txt. "
            "Default: images/timestamps.txt"
        ),
    )
    p.add_argument(
        "--calibration",
        default=None,
        help=(
            "Relative path (inside --sequence) to cam_to_cam.yaml. "
            "Default: calibration/cam_to_cam.yaml"
        ),
    )
    p.add_argument(
        "--camera-key",
        default="camRect0",
        help="Camera key in cam_to_cam.yaml intrinsics block. Default: camRect0",
    )
    p.add_argument(
        "--output",
        default="trajectory.tum",
        help="Output path for TUM-format trajectory. Default: trajectory.tum",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames (0 = run all). Useful for quick tests.",
    )
    p.add_argument(
        "--async-sba",
        action="store_true",
        default=False,
        help="Enable asynchronous sparse bundle adjustment.",
    )
    p.add_argument(
        "--no-motion-model",
        action="store_true",
        default=False,
        help=(
            "Disable the inter-frame motion model (constant-velocity pose prediction). "
            "IMPORTANT: use this for event time-surface images. "
            "The motion model predicts where LK should search for features based on "
            "estimated velocity. For RGB this helps; for sparse event images it causes "
            "LK to search in the wrong region and lose tracks, exploding RMSE. "
            "Rule of thumb: --no-motion-model for events, leave it off for RGB."
        ),
    )
    p.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration (fall back to CPU).",
    )
    p.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable Rerun visualization.",
    )
    p.add_argument(
        "--save-rrd",
        default=None,
        help="Save Rerun recording to this .rrd file.",
    )
    p.add_argument(
        "--verbosity",
        type=int,
        default=0,
        help="cuVSLAM verbosity level.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ---- Resolve paths ----------------------------------------------------
    seq = args.sequence
    if not os.path.isdir(seq):
        sys.exit(f"Error: sequence directory not found: {seq}")

    timestamps_path = (
        os.path.join(seq, args.timestamps)
        if args.timestamps
        else default_timestamps_path(seq)
    )
    image_dir = (
        os.path.join(seq, args.image_dir)
        if args.image_dir
        else default_image_dir(seq, side="left")
    )
    calibration_path = (
        os.path.join(seq, args.calibration)
        if args.calibration
        else default_calibration_path(seq)
    )

    # ---- Load dataset -----------------------------------------------------
    print(f"Loading sequence: {seq}")
    frames = build_frame_sequence(timestamps_path, image_dir)
    print(f"  {len(frames)} frames found")

    if args.max_frames > 0:
        frames = frames[: args.max_frames]
        print(f"  Capped to {len(frames)} frames (--max-frames)")

    # ---- Load calibration -------------------------------------------------
    camera = load_camera(calibration_path, args.camera_key)
    print(
        f"  Camera [{args.camera_key}]: "
        f"{camera.size[0]}×{camera.size[1]}, "
        f"focal=[{camera.focal[0]:.2f}, {camera.focal[1]:.2f}], "
        f"principal=[{camera.principal[0]:.2f}, {camera.principal[1]:.2f}]"
    )

    # ---- cuVSLAM setup ----------------------------------------------------
    cuvslam.set_verbosity(args.verbosity)

    cfg = cuvslam.Tracker.OdometryConfig(
        odometry_mode=cuvslam.Tracker.OdometryMode.Mono,
        async_sba=args.async_sba,
        use_gpu=not args.no_gpu,
        use_motion_model=not args.no_motion_model,
        enable_observations_export=True,
        enable_landmarks_export=True,
        enable_final_landmarks_export=True,
    )
    rig = cuvslam.Rig([camera])
    tracker = cuvslam.Tracker(rig, cfg)
    print(
        f"  Tracker initialized (mode=Mono, gpu={not args.no_gpu}, "
        f"motion_model={not args.no_motion_model})"
    )

    # ---- Visualization setup ----------------------------------------------
    enable_viz = not args.no_viz
    if enable_viz and not HAS_RERUN:
        print(
            "Warning: rerun-sdk is not installed. "
            "Run with --no-viz or install rerun-sdk. Disabling visualization."
        )
        enable_viz = False
    if enable_viz:
        setup_visualizer(args.save_rrd)

    # ---- Tracking loop ----------------------------------------------------
    trajectory_3d: List = []           # for rerun line strip
    tum_rows: List[List[float]] = []   # for TUM file output

    failed = 0
    for frame_id, (timestamp_ns, image_path) in enumerate(frames):
        image = load_frame(image_path)

        # -----------------------------------------------------------------
        # Extension point: event camera fusion
        #
        # When you are ready to add event frames, load the event frame here
        # and pass it as a second image in the ImageSet (camera_index=1).
        # The second camera must be registered in the rig with its own
        # cuvslam.Camera intrinsics (same resolution as the event sensor).
        #
        # Example (not active):
        #   from dsec_utils import load_event_frame
        #   event_frame = load_event_frame(seq, frame_id)   # (H, W) uint8
        #   odom_pose, _ = tracker.track(timestamp_ns, [image, event_frame])
        # -----------------------------------------------------------------
        odom_pose_estimate, _ = tracker.track(timestamp_ns, [image])

        if odom_pose_estimate.world_from_rig is None:
            failed += 1
            if failed <= 5:
                print(f"  Warning: tracking failed at frame {frame_id}")
            elif failed == 6:
                print("  (further tracking failures suppressed)")
            continue

        pose = odom_pose_estimate.world_from_rig.pose
        trajectory_3d.append(pose.translation)
        tum_rows.append(
            [timestamp_ns] + list(pose.translation) + list(pose.rotation)
        )

        if enable_viz:
            log_frame(
                frame_id=frame_id,
                pose=pose,
                image=image,
                observations=tracker.get_last_observations(0),
                landmarks=tracker.get_last_landmarks(),
                final_landmarks=tracker.get_final_landmarks(),
                trajectory=trajectory_3d,
                camera=camera,
            )

    # ---- Save output ------------------------------------------------------
    print(f"\nTracking complete: {len(tum_rows)} poses, {failed} failures")
    if tum_rows:
        save_tum(args.output, tum_rows)
    else:
        print("No poses produced. Check your calibration and image data.")


if __name__ == "__main__":
    main()
