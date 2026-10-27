"""Convert tracked XYZ points into center velocity and joint angular velocity."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MOTION_DIR = DATA_DIR / "rotato"
DATA_GLOB = "*_smoothed_coords.csv"
CENTER_POINT = 1
BODY_POINTS = (1, 2, 3, 4, 5)
LEG_CHAINS = (
    (11, 17, 23),
    (9, 15, 21),
    (7, 13, 19),
    (6, 12, 18),
    (8, 14, 20),
    (10, 16, 22),
)
JOINTS_PER_LEG = 3  # center->root, root->middle, middle->tip
JOINT_COUNT = len(LEG_CHAINS) * JOINTS_PER_LEG
FEATURE_COUNT = 3 + JOINT_COUNT * 7


def coordinate_columns():
    return [f"pt{i}_{axis}" for i in range(1, 24) for axis in "XYZ"]


def feature_columns():
    columns = ["center_vx", "center_vy", "center_vz"]
    for leg_index in range(1, len(LEG_CHAINS) + 1):
        for joint_index, joint_name in enumerate(("body_root", "upper_knee", "lower_knee"), 1):
            prefix = f"leg{leg_index}_{joint_index}_{joint_name}"
            columns.extend([
                f"{prefix}_vx", f"{prefix}_vy", f"{prefix}_vz",
                f"{prefix}_axis_x", f"{prefix}_axis_y", f"{prefix}_axis_z",
                f"{prefix}_rad_per_frame",
            ])
    return columns


def load_coordinates(path):
    frame = pd.read_csv(path, sep=None, engine="python")
    frame.columns = frame.columns.astype(str).str.strip()
    columns = coordinate_columns()
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing coordinate columns {missing[:5]}")
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: coordinates contain missing or nonnumeric values")
    return values.reshape(-1, 23, 3), columns


def canonical_basis(coordinates):
    """Put each recording in a stable body-aligned frame for shared training."""
    median = np.median(coordinates, axis=0)
    forward = 0.5 * (median[3] + median[4] - median[1] - median[2])
    side = 0.5 * (median[1] - median[2] + median[3] - median[4])
    forward /= max(float(np.linalg.norm(forward)), 1e-8)
    side -= np.dot(side, forward) * forward
    side /= max(float(np.linalg.norm(side)), 1e-8)
    up = np.cross(forward, side)
    up /= max(float(np.linalg.norm(up)), 1e-8)
    return np.stack((forward, side, up), axis=1).astype(np.float32)


def to_canonical(coordinates, origin=None, basis=None):
    basis = canonical_basis(coordinates) if basis is None else basis
    origin = coordinates[0, CENTER_POINT - 1].copy() if origin is None else origin
    return ((coordinates - origin) @ basis).astype(np.float32), origin, basis


def _axis_angle_between(previous, current):
    prev_norm = np.linalg.norm(previous)
    curr_norm = np.linalg.norm(current)
    if prev_norm < 1e-8 or curr_norm < 1e-8:
        return np.zeros(3, np.float32), 0.0
    a, b = previous / prev_norm, current / curr_norm
    cross = np.cross(a, b)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    angle = float(np.arctan2(sine, cosine))
    if sine > 1e-8:
        axis = cross / sine
    elif cosine < 0.0:
        reference = np.eye(3, dtype=np.float32)[int(np.argmin(np.abs(a)))]
        axis = np.cross(a, reference)
        axis /= max(float(np.linalg.norm(axis)), 1e-8)
    else:
        return np.zeros(3, np.float32), 0.0
    return axis.astype(np.float32), angle


def segment_vectors(coordinates):
    center = coordinates[:, CENTER_POINT - 1]
    vectors = []
    for chain in LEG_CHAINS:
        root, middle, tip = (coordinates[:, point - 1] for point in chain)
        vectors.extend((root - center, middle - root, tip - middle))
    return np.stack(vectors, axis=1).astype(np.float32)  # T x 18 x 3


def coordinates_to_features(coordinates):
    """Return one feature row per pose; angular velocities are rad/frame."""
    center = coordinates[:, CENTER_POINT - 1]
    center_velocity = np.zeros_like(center, dtype=np.float32)
    center_velocity[1:] = np.diff(center, axis=0)
    vectors = segment_vectors(coordinates)
    joints = np.zeros((len(coordinates), JOINT_COUNT, 7), dtype=np.float32)
    joint_points = np.stack([
        coordinates[:, point - 1]
        for chain in LEG_CHAINS for point in chain
    ], axis=1)
    joints[1:, :, :3] = np.diff(joint_points, axis=0)
    for frame in range(1, len(coordinates)):
        for joint in range(JOINT_COUNT):
            axis, angle = _axis_angle_between(vectors[frame - 1, joint], vectors[frame, joint])
            joints[frame, joint, 3:6] = axis
            joints[frame, joint, 6] = angle
    return np.concatenate((center_velocity, joints.reshape(len(coordinates), -1)), axis=1)


def extract_dataset(path, output_dir=MOTION_DIR):
    coordinates, _ = load_coordinates(path)
    canonical, origin, basis = to_canonical(coordinates)
    features = coordinates_to_features(canonical)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{path.stem.replace('_smoothed_coords', '')}_rotato.csv"
    pd.DataFrame(features, columns=feature_columns()).to_csv(output, index=False, float_format="%.9f")
    print(f"{path.name}: {len(coordinates)} frames -> {output}")
    return output


def main():
    paths = sorted(DATA_DIR.glob(DATA_GLOB))
    if not paths:
        raise FileNotFoundError(f"No tracked coordinate CSV files found in {DATA_DIR}")
    for path in paths:
        extract_dataset(path)


if __name__ == "__main__":
    main()
