"""Integrate center and joint velocities back into the 23-point CSV format."""

from pathlib import Path

import numpy as np

from dots_to_rotato import CENTER_POINT, LEG_CHAINS, feature_columns


# Scarites_6 median segment lengths from the tracked data, in coordinate units.
# The center->root links are new in this articulated representation.
DEFAULT_LINK_LENGTHS = np.asarray([
    [11.70, 6.87, 4.65],
    [9.86, 4.99, 3.67],
    [12.46, 7.81, 3.92],
    [12.30, 4.86, 3.84],
    [10.02, 5.36, 3.38],
    [12.14, 5.79, 5.12],
], dtype=np.float32)


def _rotate(vector, axis, angle):
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-8 or abs(float(angle)) < 1e-10:
        return vector
    axis = axis / axis_norm
    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * np.dot(axis, vector) * (1.0 - np.cos(angle))
    )


def build_state(canonical_coordinates, frame_index, origin, basis):
    """Capture the pose at the final seed frame and exact starting link lengths."""
    frame = canonical_coordinates[frame_index]
    center = frame[CENTER_POINT - 1]
    body_offsets = [frame[point - 1] - center for point in (2, 3, 4, 5)]
    vectors = []
    for chain in LEG_CHAINS:
        root, middle, tip = (frame[point - 1] for point in chain)
        vectors.extend((root - center, middle - root, tip - middle))
    vectors = np.asarray(vectors, dtype=np.float32)
    lengths = np.linalg.norm(vectors, axis=1).reshape(6, 3)
    return {
        "frame_index": int(frame_index),
        "center": center.astype(float).tolist(),
        "body_offsets": np.asarray(body_offsets, dtype=np.float32).astype(float).tolist(),
        "joint_vectors": vectors.astype(float).tolist(),
        "link_lengths": lengths.astype(float).tolist(),
        "origin": np.asarray(origin, dtype=np.float32).astype(float).tolist(),
        "basis": np.asarray(basis, dtype=np.float32).astype(float).tolist(),
        "units": "source coordinate units; angular rate is radians/frame",
    }


def integrate(canonical_state, future_features):
    """Integrate the given feature rows, keeping every link length fixed."""
    state = canonical_state
    center = np.asarray(state["center"], dtype=np.float64)
    body_offsets = np.asarray(state["body_offsets"], dtype=np.float64)
    vectors = np.asarray(state["joint_vectors"], dtype=np.float64).reshape(18, 3)
    lengths = np.asarray(state.get("link_lengths", DEFAULT_LINK_LENGTHS), dtype=np.float64).reshape(18)
    features = np.asarray(future_features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != len(feature_columns()):
        raise ValueError(f"Expected future feature matrix with {len(feature_columns())} columns")

    frames = np.empty((len(features), 23, 3), dtype=np.float32)
    for frame_index, feature in enumerate(features):
        center = center + feature[:3]
        angular = feature[3:].reshape(18, 7)
        for joint_index in range(18):
            axis = angular[joint_index, 3:6]
            rate = angular[joint_index, 6]
            vector = _rotate(vectors[joint_index], axis, rate)
            norm = np.linalg.norm(vector)
            if norm > 1e-8:
                vectors[joint_index] = vector * (lengths[joint_index] / norm)

        pose = np.zeros((23, 3), dtype=np.float64)
        pose[0] = center
        pose[1:5] = center + body_offsets
        for leg_index, chain in enumerate(LEG_CHAINS):
            root, middle, tip = chain
            base = leg_index * 3
            pose[root - 1] = center + vectors[base]
            pose[middle - 1] = pose[root - 1] + vectors[base + 1]
            pose[tip - 1] = pose[middle - 1] + vectors[base + 2]
        frames[frame_index] = pose.astype(np.float32)
    return frames


def to_source_coordinates(canonical_coordinates, origin, basis):
    return np.asarray(canonical_coordinates) @ np.asarray(basis).T + np.asarray(origin)

