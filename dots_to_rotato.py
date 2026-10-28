"""Stage 0: translate point tracks into rigid-body and relative-joint motion."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import config


BODY_COLUMNS = [
    "front_com_vx", "front_com_vy", "front_com_vz",
    "front_axis_x", "front_axis_y", "front_axis_z", "front_rad_per_frame",
    "rear_axis_x", "rear_axis_y", "rear_axis_z", "rear_rad_per_frame",
]
LEG_COLUMNS = [
    f"leg{leg}_joint{joint}_{name}"
    for leg in range(1, 7)
    for joint in range(1, 4)
    for name in ("axis_x", "axis_y", "axis_z", "rad_per_frame")
]


def coordinate_columns():
    return [f"pt{i}_{axis}" for i in range(1, 24) for axis in "XYZ"]


def load_coordinates(path):
    frame = pd.read_csv(path, sep=None, engine="python")
    frame.columns = frame.columns.astype(str).str.strip()
    columns = coordinate_columns()
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing coordinate columns {missing[:5]}")
    points = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    if not np.isfinite(points).all():
        raise ValueError(f"{path}: coordinates contain missing or nonnumeric values")
    return points.reshape(-1, 23, 3), columns


def _unit(vector, fallback=None):
    norm = np.linalg.norm(vector)
    if norm > 1e-10:
        return vector / norm
    return np.array([1.0, 0.0, 0.0]) if fallback is None else fallback


def triangle_frame(triangle):
    """Return a right-handed frame whose z axis is the triangle normal."""
    p1, p2, p3 = triangle
    x_axis = _unit(p2 - p1)
    normal = _unit(np.cross(p2 - p1, p3 - p1), np.array([0.0, 0.0, 1.0]))
    y_axis = _unit(np.cross(normal, x_axis))
    normal = _unit(np.cross(x_axis, y_axis))
    return np.column_stack((x_axis, y_axis, normal))


def canonical_basis(points):
    """Align a recording to its median body axes before pooling beetles."""
    median = np.median(points, axis=0)
    forward = 0.5 * (median[3] + median[4] - median[1] - median[2])
    side = 0.5 * (median[1] - median[2] + median[3] - median[4])
    forward = _unit(forward)
    side = _unit(side - np.dot(side, forward) * forward)
    up = _unit(np.cross(forward, side))
    return np.column_stack((forward, side, up))


def canonicalize(points):
    basis = canonical_basis(points)
    origin = points[0, 0].copy()
    return ((points - origin) @ basis), origin, basis


def _delta_axis_angle(delta_matrix):
    vector = Rotation.from_matrix(delta_matrix).as_rotvec()
    angle = np.linalg.norm(vector)
    if angle < 1e-9:
        return np.zeros(3), 0.0
    return vector / angle, float(angle)


def _frame_from_direction(direction, reference_frame):
    z_axis = _unit(direction, reference_frame[:, 2])
    x_axis = reference_frame[:, 0] - np.dot(reference_frame[:, 0], z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = reference_frame[:, 1] - np.dot(reference_frame[:, 1], z_axis) * z_axis
    x_axis = _unit(x_axis)
    y_axis = _unit(np.cross(z_axis, x_axis))
    x_axis = _unit(np.cross(y_axis, z_axis))
    return np.column_stack((x_axis, y_axis, z_axis))


def _minimal_rotation(old_direction, new_direction, fallback_axis):
    a, b = _unit(old_direction), _unit(new_direction)
    cross = np.cross(a, b)
    sine = np.linalg.norm(cross)
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if sine < 1e-9:
        if cosine > 0:
            return np.eye(3)
        axis = fallback_axis - np.dot(fallback_axis, a) * a
        axis = _unit(axis, np.array([0.0, 1.0, 0.0]))
        return Rotation.from_rotvec(axis * np.pi).as_matrix()
    axis = cross / sine
    return Rotation.from_rotvec(axis * np.arctan2(sine, cosine)).as_matrix()


def link_frames(vectors, body_frames):
    frames = np.empty((len(vectors), 3, 3), dtype=np.float64)
    frames[0] = _frame_from_direction(vectors[0], body_frames[0])
    for t in range(1, len(vectors)):
        turn = _minimal_rotation(vectors[t - 1], vectors[t], body_frames[t - 1, :, 0])
        frames[t] = turn @ frames[t - 1]
        # Remove accumulated floating-point drift and re-align the link axis.
        frames[t] = _frame_from_direction(vectors[t], frames[t])
    return frames


def rotations_to_velocity(rotations):
    values = np.zeros((len(rotations), 4), dtype=np.float64)
    for t in range(1, len(rotations)):
        delta = rotations[t] @ rotations[t - 1].T
        axis, rate = _delta_axis_angle(delta)
        values[t, :3] = axis
        values[t, 3] = rate
    return values


def translate_recording(points):
    canonical, origin, basis = canonicalize(points)
    frame_count = len(canonical)
    front_points = canonical[:, np.array(config.FRONT_TRIANGLE) - 1]
    rear_points = canonical[:, np.array(config.REAR_TRIANGLE) - 1]
    front_com = front_points.mean(axis=1)
    front_frames = np.stack([triangle_frame(row) for row in front_points])
    rear_frames = np.stack([triangle_frame(row) for row in rear_points])

    body = np.zeros((frame_count, config.BODY_FEATURES), dtype=np.float64)
    body[1:, :3] = np.diff(front_com, axis=0)
    body[:, 3:7] = rotations_to_velocity(front_frames)
    rear_relative = np.einsum("tji,tjk->tik", front_frames, rear_frames)
    body[:, 7:11] = rotations_to_velocity(rear_relative)

    legs = np.zeros((frame_count, config.LEG_FEATURES), dtype=np.float64)
    link_vectors_by_leg = []
    relative_rotations_by_leg = []
    link_lengths_by_leg = []
    virtual_joint_angles_by_leg = []
    for leg_index, chain in enumerate(config.LEG_CHAINS):
        root, middle, tip = (canonical[:, point - 1] for point in chain)
        anchor_point = config.LEG_ANCHORS[leg_index]
        body_anchor = canonical[:, anchor_point - 1]
        vectors = np.stack((root - body_anchor, middle - root, tip - middle), axis=1)
        anchor_frames = front_frames if anchor_point <= 3 else rear_frames
        frames = []
        parent_frames = anchor_frames
        for joint_index in range(3):
            child_frames = link_frames(vectors[:, joint_index], parent_frames)
            frames.append(child_frames)
            parent_frames = child_frames
        parents = (anchor_frames, frames[0], frames[1])
        relative = np.stack([
            np.einsum("tji,tjk->tik", parents[j], frames[j])
            for j in range(3)
        ], axis=1)
        for joint in range(3):
            start = (leg_index * 3 + joint) * 4
            legs[:, start:start + 4] = rotations_to_velocity(relative[:, joint])
        link_vectors_by_leg.append(vectors)
        relative_rotations_by_leg.append(relative)
        link_lengths_by_leg.append(np.linalg.norm(vectors, axis=-1))
        direction_local = relative[:, 0, :, 2]
        azimuth = np.unwrap(np.arctan2(direction_local[:, 1], direction_local[:, 0]))
        elevation = np.arctan2(
            direction_local[:, 2],
            np.linalg.norm(direction_local[:, :2], axis=-1),
        )
        virtual_joint_angles_by_leg.append(np.stack((azimuth, elevation), axis=-1))

    front_local = np.einsum(
        "tki,tpk->tpi", front_frames,
        front_points - front_com[:, None, :],
    )
    rear_local = np.einsum(
        "tki,tpk->tpi", rear_frames,
        rear_points - rear_points[:, :1, :],
    )
    rear_rel_rotation = np.einsum("tji,tjk->tik", front_frames, rear_frames)
    rear_front_normal_deviation = np.arccos(np.clip(
        np.abs(rear_rel_rotation[:, 2, 2]), 0.0, 1.0
    ))
    front_normals = front_frames[:, :, 2]
    front_normal_turns = np.arccos(np.clip(
        np.sum(front_normals[1:] * front_normals[:-1], axis=-1), -1.0, 1.0
    ))
    head_vectors = 0.5 * (front_points[:, 1] + front_points[:, 2]) - front_points[:, 0]
    head_pitch = np.arcsin(np.clip(
        np.sum(head_vectors * basis[2], axis=-1)
        / np.maximum(np.linalg.norm(head_vectors, axis=-1), 1e-12),
        -1.0, 1.0,
    ))
    morphology_samples = {
        # The input CSV's +Z direction expressed in this recording's canonical frame.
        # canonical_basis() may point its third axis opposite to world up.
        "canonical_up": basis[2].copy(),
        "belly_heights": np.einsum(
            "tpi,i->tp", canonical[:, np.array(config.REAR_TRIANGLE[1:]) - 1]
            - front_com[:, None, :], basis[2]
        ),
        "foot_heights": np.einsum(
            "tpi,i->tp", canonical[:, np.array([chain[2] for chain in config.LEG_CHAINS]) - 1]
            - front_com[:, None, :], basis[2]
        ),
        "virtual_joint_angles": np.stack(virtual_joint_angles_by_leg, axis=1),
        "front_normal_turns": front_normal_turns,
        "rear_front_normal_deviation": rear_front_normal_deviation,
        "head_pitch": head_pitch,
        "front_local": front_local,
        "rear_local": rear_local,
        "rear_relative": rear_rel_rotation,
        "leg_relative": np.concatenate(relative_rotations_by_leg, axis=1),
        "leg_lengths": np.concatenate(link_lengths_by_leg, axis=1),
        "front_edges": np.stack((
            np.linalg.norm(front_points[:, 0] - front_points[:, 1], axis=-1),
            np.linalg.norm(front_points[:, 0] - front_points[:, 2], axis=-1),
            np.linalg.norm(front_points[:, 1] - front_points[:, 2], axis=-1),
        ), axis=1),
        "rear_edges": np.stack((
            np.linalg.norm(rear_points[:, 0] - rear_points[:, 1], axis=-1),
            np.linalg.norm(rear_points[:, 0] - rear_points[:, 2], axis=-1),
            np.linalg.norm(rear_points[:, 1] - rear_points[:, 2], axis=-1),
        ), axis=1),
    }
    return body, legs, morphology_samples


def _rotation_mean(matrices):
    return Rotation.from_matrix(np.asarray(matrices)).mean().as_matrix()


def make_morphology(samples):
    front_local = np.concatenate([x["front_local"] for x in samples], axis=0)
    rear_local = np.concatenate([x["rear_local"] for x in samples], axis=0)
    rear_relative = np.concatenate([x["rear_relative"] for x in samples], axis=0)
    leg_relative = np.concatenate([x["leg_relative"] for x in samples], axis=0)
    leg_lengths = np.concatenate([x["leg_lengths"] for x in samples], axis=0)
    front_edges = np.concatenate([x["front_edges"] for x in samples], axis=0)
    rear_edges = np.concatenate([x["rear_edges"] for x in samples], axis=0)
    canonical_up = np.mean([x["canonical_up"] for x in samples], axis=0)
    canonical_up /= max(np.linalg.norm(canonical_up), 1e-12)
    belly_heights = np.concatenate([x["belly_heights"].reshape(-1) for x in samples])
    foot_heights = np.concatenate([x["foot_heights"].reshape(-1) for x in samples])
    virtual_angles = [x["virtual_joint_angles"] for x in samples]
    # Unwrapped azimuths from separate recordings may differ by whole turns.
    # Align each recording to a shared circular center before pooling limits.
    circular_center = np.arctan2(
        np.mean([np.sin(x).mean(axis=0) for x in virtual_angles], axis=0),
        np.mean([np.cos(x).mean(axis=0) for x in virtual_angles], axis=0),
    )
    aligned_virtual_angles = []
    for angle_values in virtual_angles:
        aligned = angle_values.copy()
        for leg_index in range(6):
            azimuth = aligned[:, leg_index, 0]
            shift = 2.0 * np.pi * np.round(
                (circular_center[leg_index, 0] - np.median(azimuth)) / (2.0 * np.pi)
            )
            aligned[:, leg_index, 0] += shift
        aligned_virtual_angles.append(aligned)
    virtual_angles = np.concatenate(aligned_virtual_angles, axis=0)
    front_normal_turns = np.concatenate([x["front_normal_turns"] for x in samples])
    rear_front_normal_deviation = np.concatenate([
        x["rear_front_normal_deviation"] for x in samples
    ])
    head_pitch = np.concatenate([x["head_pitch"] for x in samples])
    # Stage 3 uses a consistent +Z-up output frame. The canonical frame's vertical
    # direction is learned from the original CSV Z axis, then mapped to output +Z.
    canonical_to_world = Rotation.align_vectors(
        np.array([[0.0, 0.0, 1.0]]), canonical_up[None, :]
    )[0].as_matrix()

    all_angular = []
    all_acceleration = []
    for body, legs in zip(
        [x["body"] for x in samples], [x["legs"] for x in samples]
    ):
        body_omega = body[:, [3, 4, 5, 7, 8, 9]]
        body_rate = body[:, [6, 10]]
        angular = np.concatenate((
            np.concatenate((body_omega[:, :3], body_rate[:, :1]), axis=1),
            np.concatenate((body_omega[:, 3:], body_rate[:, 1:]), axis=1),
            legs.reshape(len(legs), 18, 4).reshape(len(legs), -1),
        ), axis=1)
        all_angular.append(angular)
        omega = angular.reshape(len(angular), -1, 4)
        omega_vectors = omega[:, :, :3] * omega[:, :, 3:4]
        all_acceleration.append(np.linalg.norm(np.diff(omega_vectors, axis=0), axis=-1).reshape(-1))
    angular = np.concatenate(all_angular, axis=0)
    rates = np.concatenate((angular[:, 3::4],), axis=1)
    accel = np.concatenate(all_acceleration)

    return {
        "canonical_up": canonical_up.tolist(),
        "canonical_to_world_rotation": canonical_to_world.tolist(),
        "belly_min_height": float(np.percentile(belly_heights, 1.0)),
        "foot_max_height": float(np.percentile(foot_heights, 99.0)),
        "height_limits_percentiles": {"belly_min": 1.0, "foot_max": 99.0},
        "virtual_joint_angle_limits": [
            {
                "azimuth_min_rad": float(np.min(virtual_angles[:, leg, 0])),
                "azimuth_max_rad": float(np.max(virtual_angles[:, leg, 0])),
                "elevation_min_rad": float(np.min(virtual_angles[:, leg, 1])),
                "elevation_max_rad": float(np.max(virtual_angles[:, leg, 1])),
            }
            for leg in range(6)
        ],
        "max_front_normal_turn_rad_per_frame": float(np.max(front_normal_turns)),
        "max_rear_front_normal_deviation_rad": float(np.max(rear_front_normal_deviation)),
        "max_head_tilt_rad": float(np.percentile(np.abs(head_pitch), 95.0)),
        "head_tilt_limit_percentile": 95.0,
        "front_local_points": np.median(front_local, axis=0).tolist(),
        "rear_local_points_from_pt1": np.median(rear_local, axis=0).tolist(),
        "rear_initial_relative_rotation": _rotation_mean(rear_relative).tolist(),
        "leg_initial_relative_rotations": [
            _rotation_mean(leg_relative[:, j]).tolist() for j in range(18)
        ],
        "leg_lengths": np.median(leg_lengths, axis=0).reshape(6, 3).tolist(),
        "front_triangle_edges": np.median(front_edges, axis=0).tolist(),
        "rear_triangle_edges": np.median(rear_edges, axis=0).tolist(),
        "max_angular_speed": max(
            config.DEFAULT_MAX_ANGULAR_SPEED,
            float(np.percentile(np.abs(rates), 99.5)) * config.ANGULAR_SPEED_MARGIN,
        ),
        "max_angular_acceleration": max(
            config.DEFAULT_MAX_ANGULAR_ACCELERATION,
            float(np.percentile(accel, 99.5)) * config.ANGULAR_ACCELERATION_MARGIN,
        ),
        "angular_rate_percentile": 99.5,
        "angular_margin": config.ANGULAR_SPEED_MARGIN,
    }


def main():
    config.TRANSLATE_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(config.DATA_DIR.glob(config.SOURCE_GLOB))
    if len(paths) < 2:
        raise FileNotFoundError(f"Need at least two source datasets in {config.DATA_DIR}")
    samples = []
    for path in paths:
        points, _ = load_coordinates(path)
        body, legs, morphology = translate_recording(points)
        morphology["body"] = body
        morphology["legs"] = legs
        name = path.stem.replace("_smoothed_coords", "")
        pd.DataFrame(body, columns=BODY_COLUMNS).to_csv(
            config.TRANSLATE_DIR / f"{name}_body.csv", index=False, float_format="%.9f"
        )
        pd.DataFrame(legs, columns=LEG_COLUMNS).to_csv(
            config.TRANSLATE_DIR / f"{name}_legs.csv", index=False, float_format="%.9f"
        )
        samples.append(morphology)
        print(f"translated {path.name}: {len(points)} frames")

    morphology = make_morphology(samples)
    morphology["source_datasets"] = [p.name for p in paths]
    config.TRANSLATE_DIR.joinpath("morphology.json").write_text(
        json.dumps(morphology, indent=2), encoding="utf-8"
    )
    print(
        f"Rigid morphology and motion caps saved; omega cap="
        f"{morphology['max_angular_speed']:.4f} rad/frame, "
        f"acceleration cap={morphology['max_angular_acceleration']:.4f} rad/frame²"
    )


if __name__ == "__main__":
    main()
