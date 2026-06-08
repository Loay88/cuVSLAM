"""
DSEC dataset utilities for cuVSLAM.

Expected sequence layout (standard DSEC distribution):

    <sequence>/
    ├── images/
    │   ├── left/
    │   │   └── rectified/   ← rectified RGB frames: 000000.png, 000001.png, ...
    │   ├── right/
    │   │   └── rectified/
    │   └── timestamps.txt   ← one integer per line, microseconds, no header
    └── calibration/
        └── cam_to_cam.yaml

cam_to_cam.yaml expected layout:

    intrinsics:
      camRect0:
        camera_matrix: [fx, fy, cx, cy]
        resolution: [width, height]
        distortion_model: none       # rectified → no distortion
        distortion_coeffs: []
      camRect1:
        camera_matrix: [fx, fy, cx, cy]
        resolution: [width, height]
        distortion_model: none
        distortion_coeffs: []
    extrinsics:
      T_10:                          # 4×4 row-major list: cam0 coords → cam1 coords
        - [r00, r01, r02, tx]
        - [r10, r11, r12, ty]
        - [r20, r21, r22, tz]
        - [0.0, 0.0, 0.0, 1.0]
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml
from PIL import Image

import cuvslam

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# A frame is a (timestamp_ns, image_path) pair.
Frame = Tuple[int, str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resolve(sequence_path: str, relative: str) -> str:
    return os.path.normpath(os.path.join(sequence_path, relative))


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

def _is_csv_with_header(first_line: str) -> bool:
    """Return True if the line looks like a CSV header (non-numeric first field)."""
    first_field = first_line.split(",")[0].strip()
    try:
        int(first_field)
        return False
    except ValueError:
        return True


def load_timestamps(
    timestamps_path: str,
) -> Tuple[List[int], Optional[List[str]]]:
    """
    Read a DSEC timestamps file in either supported format.

    Supported formats
    -----------------
    Plain .txt — one integer per line (microseconds), no header::

        58391107532
        58391157545
        ...

    CSV with header — ``timestamp,filename`` columns::

        timestamp,filename
        58391107532,frame_000000.png
        58391157545,frame_000001.png
        ...

    Returns
    -------
    timestamps_ns : List[int]
        Timestamps converted from microseconds to nanoseconds.
    filenames : List[str] or None
        Filenames from the CSV ``filename`` column, or ``None`` for plain .txt.
    """
    if not os.path.exists(timestamps_path):
        raise FileNotFoundError(f"Timestamps file not found: {timestamps_path}")

    with open(timestamps_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines:
        raise ValueError(f"No data found in {timestamps_path}")

    if _is_csv_with_header(lines[0]):
        # CSV format: skip header row, parse timestamp + optional filename columns
        timestamps_ns: List[int] = []
        filenames: List[str] = []
        header_cols = [c.strip().lower() for c in lines[0].split(",")]
        has_filename_col = len(header_cols) >= 2

        for lineno, line in enumerate(lines[1:], start=2):
            parts = line.split(",")
            try:
                timestamps_ns.append(int(parts[0].strip()) * 1000)  # µs → ns
            except (ValueError, IndexError):
                raise ValueError(
                    f"Cannot parse timestamp on line {lineno} of {timestamps_path}: {line!r}"
                )
            if has_filename_col and len(parts) >= 2:
                filenames.append(parts[1].strip())

        return timestamps_ns, (filenames if has_filename_col else None)

    else:
        # Plain .txt format: one integer per line
        timestamps_ns = []
        for lineno, line in enumerate(lines, start=1):
            try:
                timestamps_ns.append(int(line) * 1000)  # µs → ns
            except ValueError:
                raise ValueError(
                    f"Non-integer value on line {lineno} of {timestamps_path}: {line!r}"
                )
        return timestamps_ns, None


# ---------------------------------------------------------------------------
# Image paths
# ---------------------------------------------------------------------------

def load_image_paths(image_dir: str) -> List[str]:
    """
    Return sorted list of .png paths from image_dir.

    DSEC images are named 000000.png, 000001.png, etc.
    """
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    paths = sorted(Path(image_dir).glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No .png files found in: {image_dir}")

    return [str(p) for p in paths]


# ---------------------------------------------------------------------------
# Frame sequence
# ---------------------------------------------------------------------------

def build_frame_sequence(timestamps_path: str, image_dir: str) -> List[Frame]:
    """
    Build a sorted list of (timestamp_ns, absolute_image_path) pairs.

    Accepts both timestamp file formats:

    - **Plain .txt**: timestamps only → image paths are globbed from image_dir
      in sorted order and zipped with timestamps. Counts must match.
    - **CSV with filename column**: filenames are taken directly from the CSV
      and joined onto image_dir. No glob needed, no count mismatch possible.
    """
    timestamps_ns, filenames = load_timestamps(timestamps_path)

    if filenames is not None:
        # CSV provided filenames — join them onto image_dir directly
        paths = [os.path.join(image_dir, f) for f in filenames]
    else:
        # Plain .txt — glob and zip; counts must match
        paths = load_image_paths(image_dir)
        if len(timestamps_ns) != len(paths):
            raise ValueError(
                f"Timestamp/image count mismatch: "
                f"{len(timestamps_ns)} timestamps vs {len(paths)} images in {image_dir}. "
                f"If your timestamps file does not include filenames, "
                f"the image directory must contain exactly one image per timestamp."
            )

    return list(zip(timestamps_ns, paths))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def load_camera(cam_to_cam_yaml: str, camera_key: str = "camRect0") -> cuvslam.Camera:
    """
    Load a single camera from a DSEC cam_to_cam.yaml.

    Args:
        cam_to_cam_yaml: Path to calibration YAML.
        camera_key: Key under 'intrinsics', e.g. 'camRect0' or 'camRect1'.
    """
    config = _load_yaml(cam_to_cam_yaml)

    intrinsics = config.get("intrinsics", {})
    if camera_key not in intrinsics:
        available = list(intrinsics.keys())
        raise KeyError(
            f"Camera key '{camera_key}' not found in {cam_to_cam_yaml}. "
            f"Available keys: {available}"
        )

    cam_cfg = intrinsics[camera_key]

    required = {"camera_matrix", "resolution"}
    missing = required - cam_cfg.keys()
    if missing:
        raise KeyError(f"Missing fields {missing} for '{camera_key}' in {cam_to_cam_yaml}")

    fx, fy, cx, cy = cam_cfg["camera_matrix"]
    width, height = cam_cfg["resolution"]

    camera = cuvslam.Camera()
    camera.focal = [fx, fy]
    camera.principal = [cx, cy]
    camera.size = [width, height]

    distortion_model = str(cam_cfg.get("distortion_model", "none")).lower()
    coeffs = cam_cfg.get("distortion_coeffs", [])

    if distortion_model in {"none", "pinhole"} or not coeffs:
        pass  # no distortion; Camera() default is pinhole
    elif distortion_model == "fisheye":
        if len(coeffs) != 4:
            raise ValueError(
                f"Fisheye distortion expects 4 coefficients, got {len(coeffs)} "
                f"for '{camera_key}' in {cam_to_cam_yaml}"
            )
        camera.distortion = cuvslam.Distortion(cuvslam.Distortion.Model.Fisheye, coeffs)
    elif distortion_model in {"brown", "radtan", "plumb_bob"}:
        if len(coeffs) == 4:
            # k1, k2, p1, p2 → pad k3=0 to match Brown 5-param (k1, k2, k3, p1, p2)
            coeffs = [coeffs[0], coeffs[1], 0.0, coeffs[2], coeffs[3]]
        elif len(coeffs) != 5:
            raise ValueError(
                f"Brown distortion expects 4 or 5 coefficients, got {len(coeffs)} "
                f"for '{camera_key}' in {cam_to_cam_yaml}"
            )
        camera.distortion = cuvslam.Distortion(cuvslam.Distortion.Model.Brown, coeffs)
    else:
        raise ValueError(
            f"Unsupported distortion model '{distortion_model}' "
            f"for '{camera_key}' in {cam_to_cam_yaml}"
        )

    return camera


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_frame(path: str) -> np.ndarray:
    """
    Load a single image as a numpy array compatible with cuVSLAM.

    Grayscale (L) → (H, W) uint8
    RGB            → (H, W, 3) uint8 in BGR order
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = Image.open(path)
    arr = np.array(img)

    if img.mode == "L":
        return arr  # (H, W) uint8
    if img.mode == "RGB":
        return np.ascontiguousarray(arr[:, :, ::-1])  # RGB → BGR
    raise ValueError(
        f"Unsupported image mode '{img.mode}' in {path}. Expected 'L' or 'RGB'."
    )


# ---------------------------------------------------------------------------
# Extension point: event frames
# ---------------------------------------------------------------------------

def load_event_frame(sequence_path: str, frame_idx: int) -> np.ndarray:
    """
    Load a pre-rendered event frame (time surface or SAE) for frame_idx.

    NOT YET IMPLEMENTED. This is the extension point for RGB+event fusion.

    When implemented, this function should:
    1. Read the event stream from 'events/left/events.h5' (DSEC format).
    2. Accumulate events in the time window [t_{frame_idx-1}, t_{frame_idx}].
    3. Render a time-surface or SAE as a (H, W) uint8 image.
    4. Return that image so the caller can pass it to a second IMonoSOF instance.

    Raises:
        NotImplementedError: Always, until implemented.
    """
    raise NotImplementedError(
        "Event frame loading is not yet implemented. "
        "Implement event-to-frame conversion here before enabling event fusion."
    )


# ---------------------------------------------------------------------------
# Standard DSEC path helpers
# ---------------------------------------------------------------------------

def default_timestamps_path(sequence_path: str) -> str:
    return _resolve(sequence_path, "images/timestamps.txt")


def default_image_dir(sequence_path: str, side: str = "left") -> str:
    return _resolve(sequence_path, f"images/{side}/rectified")


def default_calibration_path(sequence_path: str) -> str:
    return _resolve(sequence_path, "calibration/cam_to_cam.yaml")
