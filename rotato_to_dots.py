"""Stage 3: reconstruct points from body and relative-leg angular velocities."""

import json

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import config
from dots_to_rotato import (
    BODY_COLUMNS, LEG_COLUMNS, canonicalize, coordinate_columns, link_frames,
    load_coordinates, triangle_frame,
)


def _triangle_points(edges):
    d12, d13, d23 = map(float, edges)
    x = (d12 * d12 + d13 * d13 - d23 * d23) / (2.0 * d12)
    y = np.sqrt(max(d13 * d13 - x * x, 1e-8))
    triangle = np.array([[0.0, 0.0, 0.0], [d12, 0.0, 0.0], [x, y, 0.0]])
    return triangle - triangle.mean(axis=0)


def _fallback_morphology():
    front = _triangle_points(config.DEFAULT_FRONT_TRIANGLE_EDGES)
    rear = np.array([
        [0.0, 0.0, 0.0],
        [config.DEFAULT_REAR_TRIANGLE_EDGES[0], 0.0, 0.0],
        [
            (config.DEFAULT_REAR_TRIANGLE_EDGES[0] ** 2
             + config.DEFAULT_REAR_TRIANGLE_EDGES[1] ** 2
             - config.DEFAULT_REAR_TRIANGLE_EDGES[2] ** 2)
            / (2.0 * config.DEFAULT_REAR_TRIANGLE_EDGES[0]),
            0.0,
            0.0,
        ],
    ])
    x = rear[2, 0]
    rear[2, 1] = np.sqrt(max(config.DEFAULT_REAR_TRIANGLE_EDGES[1] ** 2 - x * x, 1e-8))
    return {
        "front_local_points": front.tolist(),
        "rear_local_points_from_pt1": rear.tolist(),
        "rear_initial_relative_rotation": np.eye(3).tolist(),
        "leg_initial_relative_rotations": [np.eye(3).tolist() for _ in range(config.JOINT_COUNT)],
        "leg_lengths": config.DEFAULT_LEG_LENGTHS,
    }


def _rotate_by_rate(rotation, axis, rate):
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-9 or abs(float(rate)) < 1e-10:
        return rotation
    delta = Rotation.from_rotvec(axis / norm * float(rate)).as_matrix()
    return delta @ rotation


def _rotation_between(source, target):
    source = source / max(np.linalg.norm(source), 1e-12)
    target = target / max(np.linalg.norm(target), 1e-12)
    cross = np.cross(source, target)
    sine = np.linalg.norm(cross)
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine < 1e-10:
        if cosine > 0:
            return np.eye(3)
        fallback = np.cross(source, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(fallback) < 1e-8:
            fallback = np.cross(source, np.array([0.0, 1.0, 0.0]))
        return Rotation.from_rotvec(fallback / np.linalg.norm(fallback) * np.pi).as_matrix()
    return Rotation.from_rotvec(
        cross / sine * np.arctan2(sine, cosine)
    ).as_matrix()


def _clamp_rotation_from_seed(rotation, seed_rotation, maximum_degrees):
    delta = rotation @ seed_rotation.T
    vector = Rotation.from_matrix(delta).as_rotvec()
    angle = np.linalg.norm(vector)
    maximum = np.deg2rad(maximum_degrees)
    if angle <= maximum or angle < 1e-10:
        return rotation
    return Rotation.from_rotvec(vector * (maximum / angle)).as_matrix() @ seed_rotation


def _clamp_rear_normal(front_rotation, rear_relative):
    """Limit rear/front triangle normal-axis deviation without deforming either triangle."""
    maximum = np.deg2rad(config.REAR_NORMAL_PARALLEL_MAX_DEG)
    front_normal = front_rotation[:, 2]
    rear_rotation = front_rotation @ rear_relative
    rear_normal = rear_rotation[:, 2]
    signed_dot = float(np.clip(np.dot(front_normal, rear_normal), -1.0, 1.0))
    deviation = np.arccos(abs(signed_dot))
    if deviation <= maximum:
        return rear_relative

    reference = front_normal if signed_dot >= 0.0 else -front_normal
    tangent = rear_normal - np.dot(rear_normal, reference) * reference
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm < 1e-9:
        tangent = np.cross(reference, front_rotation[:, 0])
        tangent_norm = np.linalg.norm(tangent)
    tangent /= max(tangent_norm, 1e-12)
    target_normal = np.cos(maximum) * reference + np.sin(maximum) * tangent
    correction = _rotation_between(rear_normal, target_normal)
    corrected_rear = correction @ rear_rotation
    return front_rotation.T @ corrected_rear


def reconstruct(body_motion, leg_motion, morphology, initial_state=None):
    if len(body_motion) != len(leg_motion):
        raise ValueError("Generated body and leg motion lengths differ")
    front_local = np.asarray(morphology["front_local_points"], dtype=np.float64)
    rear_local = np.asarray(morphology["rear_local_points_from_pt1"], dtype=np.float64)
    leg_lengths = np.asarray(morphology["leg_lengths"], dtype=np.float64).reshape(6, 3)
    canonical_up = np.asarray(morphology.get("canonical_up", (0.0, 0.0, 1.0)), dtype=np.float64)
    foot_max_height = morphology.get("foot_max_height")
    if initial_state is None:
        front_rotation = np.eye(3)
        rear_relative = np.asarray(morphology["rear_initial_relative_rotation"], dtype=np.float64)
        leg_relative = np.asarray(
            morphology["leg_initial_relative_rotations"], dtype=np.float64
        ).reshape(6, 3, 3, 3)
        front_com = np.zeros(3, dtype=np.float64)
        output_rotation = np.asarray(
            morphology.get("canonical_to_world_rotation", np.eye(3)), dtype=np.float64
        )
        output_offset = np.zeros(3, dtype=np.float64)
    else:
        front_local = initial_state["front_local"]
        rear_local = initial_state["rear_local"]
        leg_lengths = initial_state["leg_lengths"]
        front_rotation = initial_state["front_rotation"]
        rear_relative = initial_state["rear_relative"]
        leg_relative = initial_state["leg_relative"].copy()
        front_com = initial_state["front_com"].copy()
        # canonicalize uses row vectors: canonical = (world - origin) @ basis.
        # For column vectors in the reconstruction loop, the inverse is
        # world = basis @ canonical + origin (not basis.T).
        output_rotation = initial_state["basis"]
        output_offset = initial_state["origin"]
    front_rotation_seed = front_rotation.copy()
    rear_relative_seed = rear_relative.copy()

    points = np.zeros((len(body_motion), 23, 3), dtype=np.float64)
    for t, (body_row, legs_row) in enumerate(zip(body_motion, leg_motion)):
        front_com += body_row[:3]
        front_rotation = _rotate_by_rate(front_rotation, body_row[3:6], body_row[6])
        front_rotation = _clamp_rotation_from_seed(
            front_rotation, front_rotation_seed, config.MAX_FRONT_ROTATION_FROM_SEED_DEG
        )
        rear_relative = _rotate_by_rate(rear_relative, body_row[7:10], body_row[10])
        rear_relative = _clamp_rotation_from_seed(
            rear_relative, rear_relative_seed, config.MAX_REAR_ROTATION_FROM_FRONT_SEED_DEG
        )
        rear_relative = _clamp_rear_normal(front_rotation, rear_relative)

        front_world = front_com + front_local @ front_rotation.T
        points[t, np.array(config.FRONT_TRIANGLE) - 1] = front_world
        shared_pt1 = points[t, 0]
        rear_rotation = front_rotation @ rear_relative
        points[t, 3] = shared_pt1 + rear_rotation @ rear_local[1]
        points[t, 4] = shared_pt1 + rear_rotation @ rear_local[2]

        leg_values = legs_row.reshape(6, 3, 4)
        for leg_index, chain in enumerate(config.LEG_CHAINS):
            relative_frames = leg_relative[leg_index]
            anchor_point = config.LEG_ANCHORS[leg_index]
            anchor_rotation = front_rotation if anchor_point <= 3 else rear_rotation
            parent_rotation = anchor_rotation
            anchor = points[t, anchor_point - 1]
            positions = []
            for joint_index in range(3):
                axis, rate = leg_values[leg_index, joint_index, :3], leg_values[leg_index, joint_index, 3]
                relative_frames[joint_index] = _rotate_by_rate(
                    relative_frames[joint_index], axis, rate
                )
                child_rotation = parent_rotation @ relative_frames[joint_index]
                link = child_rotation[:, 2] * leg_lengths[leg_index, joint_index]
                positions.append(link)
                parent_rotation = child_rotation

            # The autoregressive angular velocities can drift over long rollouts.
            # If the distal point rises above the learned gait envelope, rotate
            # the complete leg around its hip. This preserves all three segment
            # lengths and the knee angles while correcting the root joint only.
            if foot_max_height is not None:
                offset = np.sum(positions, axis=0)
                offset_length = np.linalg.norm(offset)
                anchor_relative = anchor - front_com
                foot_height = float(np.dot(anchor_relative + offset, canonical_up))
                if offset_length > 1e-9 and foot_height > float(foot_max_height):
                    requested_vertical = float(foot_max_height) - float(np.dot(anchor_relative, canonical_up))
                    vertical = float(np.clip(requested_vertical, -offset_length, offset_length))
                    horizontal = offset - np.dot(offset, canonical_up) * canonical_up
                    horizontal_norm = np.linalg.norm(horizontal)
                    if horizontal_norm < 1e-9:
                        reference_axis = np.eye(3)[
                            np.argmin(np.abs(canonical_up))
                        ]
                        horizontal = np.cross(canonical_up, reference_axis)
                        horizontal_norm = np.linalg.norm(horizontal)
                    horizontal /= max(horizontal_norm, 1e-12)
                    target_offset = (
                        horizontal * np.sqrt(max(offset_length**2 - vertical**2, 0.0))
                        + canonical_up * vertical
                    )
                    correction = _rotation_between(offset, target_offset)
                    relative_frames[0] = (
                        anchor_rotation.T @ correction @ anchor_rotation @ relative_frames[0]
                    )
                    parent_rotation = anchor_rotation
                    positions = []
                    for joint_index in range(3):
                        child_rotation = parent_rotation @ relative_frames[joint_index]
                        positions.append(child_rotation[:, 2] * leg_lengths[leg_index, joint_index])
                        parent_rotation = child_rotation
            root, middle, tip = chain
            points[t, root - 1] = anchor + positions[0]
            points[t, middle - 1] = points[t, root - 1] + positions[1]
            points[t, tip - 1] = points[t, middle - 1] + positions[2]
    # With a source seed, return to that source's original coordinate system;
    # the fallback path uses the aggregate +Z-up scene frame.
    return np.einsum("ij,tpj->tpi", output_rotation, points) + output_offset


def _initial_state_from_source(path, frame_index):
    """Recover the articulated seed pose and original CSV coordinate frame."""
    points, _ = load_coordinates(path)
    if not len(points):
        raise ValueError(f"Seed source has no frames: {path}")
    frame_index = int(np.clip(frame_index, 0, len(points) - 1))
    canonical, origin, basis = canonicalize(points)
    front_points = canonical[:, np.asarray(config.FRONT_TRIANGLE) - 1]
    rear_points = canonical[:, np.asarray(config.REAR_TRIANGLE) - 1]
    front_com = front_points.mean(axis=1)
    front_frames = np.stack([triangle_frame(row) for row in front_points])
    rear_frames = np.stack([triangle_frame(row) for row in rear_points])

    t = frame_index
    front_local = (front_points[t] - front_com[t]) @ front_frames[t]
    rear_local = (rear_points[t] - rear_points[t, :1]) @ rear_frames[t]
    rear_relative = front_frames[t].T @ rear_frames[t]
    leg_relative, leg_lengths = [], []
    for leg_index, chain in enumerate(config.LEG_CHAINS):
        anchor_id = config.LEG_ANCHORS[leg_index]
        anchor_frames = front_frames if anchor_id <= 3 else rear_frames
        root, middle, tip = (canonical[:, point - 1] for point in chain)
        anchor = canonical[:, anchor_id - 1]
        vectors = np.stack((root - anchor, middle - root, tip - middle), axis=1)
        frames = []
        parent_frames = anchor_frames
        for joint_index in range(3):
            child_frames = link_frames(vectors[:, joint_index], parent_frames)
            frames.append(child_frames)
            parent_frames = child_frames
        for joint_index in range(3):
            parent = anchor_frames if joint_index == 0 else frames[joint_index - 1]
            leg_relative.append(parent[t].T @ frames[joint_index][t])
        leg_lengths.append(np.linalg.norm(vectors[t], axis=-1))

    return {
        "front_local": front_local,
        "rear_local": rear_local,
        "front_rotation": front_frames[t],
        "rear_relative": rear_relative,
        "leg_relative": np.asarray(leg_relative).reshape(6, 3, 3, 3),
        "leg_lengths": np.asarray(leg_lengths),
        "front_com": front_com[t],
        "basis": basis,
        "origin": origin,
    }


def main():
    manifest_path = config.GENERATION_DIR / "latest_run.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run stage 2 first; generation/latest_run.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    morphology_path = config.TRANSLATE_DIR / "morphology.json"
    morphology = (
        json.loads(morphology_path.read_text(encoding="utf-8"))
        if morphology_path.exists() else _fallback_morphology()
    )
    body_path = config.GENERATION_DIR / manifest["body_file"]
    legs_path = config.GENERATION_DIR / manifest["legs_file"]
    body = pd.read_csv(body_path)[BODY_COLUMNS].to_numpy(np.float64)
    legs = pd.read_csv(legs_path)[LEG_COLUMNS].to_numpy(np.float64)
    seed_file = manifest.get("seed_dataset", f"{config.TARGET_NAME}_smoothed_coords.csv")
    seed_path = config.DATA_DIR / seed_file
    seed_frame = int(manifest.get("seed_frame", config.HISTORY_N - 1))
    initial_state = _initial_state_from_source(seed_path, seed_frame)
    points = reconstruct(body, legs, morphology, initial_state=initial_state)
    config.OUTPUT_DOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.OUTPUT_DOTS_DIR / f"{manifest['run_name']}.csv"
    pd.DataFrame(points.reshape(-1, 69), columns=coordinate_columns()).to_csv(
        output_path, index=False, float_format="%.8f"
    )
    print(
        f"Built {len(points)} generated frames from {seed_file} frame {seed_frame}: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
