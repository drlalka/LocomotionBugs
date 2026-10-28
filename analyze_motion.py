"""Analyze one source CSV and its translated stage-0 motion files."""

import argparse
from pathlib import Path
from pprint import pformat

import numpy as np
import pandas as pd

import config
from dots_to_rotato import (
    BODY_COLUMNS, LEG_COLUMNS, coordinate_columns, rotations_to_velocity, triangle_frame,
)


BODY_EDGES = (
    (1, 2), (1, 3), (2, 3),
    (1, 4), (1, 5), (4, 5),
)


def _angle(a, vertex, c):
    left = a - vertex
    right = c - vertex
    left /= max(float(np.linalg.norm(left)), 1e-12)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    return float(np.degrees(np.arccos(np.clip(np.dot(left, right), -1.0, 1.0))))


def _angle_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "observed_min_deg": float(np.min(values)),
        "observed_max_deg": float(np.max(values)),
        "p01_deg": float(np.percentile(values, 1)),
        "p99_deg": float(np.percentile(values, 99)),
    }


def analyze(data_file):
    if not data_file.is_absolute():
        data_file = config.DATA_DIR / data_file
    if not data_file.exists():
        raise FileNotFoundError(data_file)
    name = data_file.stem.replace("_smoothed_coords", "")
    body_file = config.TRANSLATE_DIR / f"{name}_body.csv"
    legs_file = config.TRANSLATE_DIR / f"{name}_legs.csv"
    if not body_file.exists() or not legs_file.exists():
        raise FileNotFoundError(
            f"Translated files missing for {name}; run stage 0 before analysis"
        )

    frame = pd.read_csv(data_file, sep=None, engine="python")
    frame.columns = frame.columns.astype(str).str.strip()
    points = frame[coordinate_columns()].apply(pd.to_numeric, errors="coerce").to_numpy()
    points = points.reshape(-1, 23, 3)
    body = pd.read_csv(body_file)[BODY_COLUMNS].to_numpy(np.float64)
    legs = pd.read_csv(legs_file)[LEG_COLUMNS].to_numpy(np.float64)
    if len(points) != len(body) or len(points) != len(legs):
        raise ValueError("Source and translated files have mismatched frame counts")

    edges = list(BODY_EDGES)
    for anchor, chain in zip(config.LEG_ANCHORS, config.LEG_CHAINS):
        edges.extend(((anchor, chain[0]), (chain[0], chain[1]), (chain[1], chain[2])))
    edge_rms, edge_mse = {}, {}
    for first, second in edges:
        lengths = np.linalg.norm(points[:, first - 1] - points[:, second - 1], axis=1)
        key = f"pt{first}-pt{second}"
        edge_rms[key] = float(np.sqrt(np.mean(lengths ** 2)))
        edge_mse[key] = float(np.mean((lengths - np.mean(lengths)) ** 2))

    angular_speed = {
        "front_triangle_vs_ground": {
            "max_rad_per_frame": float(np.max(np.abs(body[:, 6]))),
            "rms_rad_per_frame": float(np.sqrt(np.mean(body[:, 6] ** 2))),
        },
        "rear_triangle_vs_front": {
            "max_rad_per_frame": float(np.max(np.abs(body[:, 10]))),
            "rms_rad_per_frame": float(np.sqrt(np.mean(body[:, 10] ** 2))),
        },
    }
    front_frames = np.stack([
        triangle_frame(points[t, np.asarray(config.FRONT_TRIANGLE) - 1])
        for t in range(len(points))
    ])
    rear_frames = np.stack([
        triangle_frame(points[t, np.asarray(config.REAR_TRIANGLE) - 1])
        for t in range(len(points))
    ])
    front_normals = front_frames[:, :, 2]
    front_normal_turns = np.degrees(np.arccos(np.clip(
        np.sum(front_normals[1:] * front_normals[:-1], axis=-1), -1.0, 1.0
    )))
    front_normal_turn_limits = {
        "min_deg_per_frame": float(np.min(front_normal_turns)),
        "max_deg_per_frame": float(np.max(front_normal_turns)),
        "p99_deg_per_frame": float(np.percentile(front_normal_turns, 99)),
    }
    rear_front_normal_deviation = np.degrees(np.arccos(np.clip(
        np.abs(np.sum(front_normals * rear_frames[:, :, 2], axis=-1)), 0.0, 1.0
    )))
    rear_front_normal_limits = {
        "min_deg": float(np.min(rear_front_normal_deviation)),
        "max_deg": float(np.max(rear_front_normal_deviation)),
        "p95_deg": float(np.percentile(rear_front_normal_deviation, 95)),
    }
    head_vectors = 0.5 * (points[:, 1] + points[:, 2]) - points[:, 0]
    head_pitch = np.degrees(np.arcsin(np.clip(
        head_vectors[:, 2] / np.maximum(np.linalg.norm(head_vectors, axis=-1), 1e-12),
        -1.0, 1.0,
    )))
    head_pitch_summary = {
        "min_deg": float(np.min(head_pitch)),
        "max_deg": float(np.max(head_pitch)),
        "p95_absolute_deg": float(np.percentile(np.abs(head_pitch), 95)),
    }
    rear_absolute_rate = rotations_to_velocity(rear_frames)[:, 3]
    front_absolute_rate = rotations_to_velocity(front_frames)[:, 3]
    angular_speed["front_triangle_vs_ground_from_source"] = {
        "max_rad_per_frame": float(np.max(np.abs(front_absolute_rate))),
        "rms_rad_per_frame": float(np.sqrt(np.mean(front_absolute_rate ** 2))),
    }
    angular_speed["rear_triangle_vs_ground_from_source"] = {
        "max_rad_per_frame": float(np.max(np.abs(rear_absolute_rate))),
        "rms_rad_per_frame": float(np.sqrt(np.mean(rear_absolute_rate ** 2))),
    }
    leg_motion = legs.reshape(len(legs), config.JOINT_COUNT, 4)
    for leg_index in range(6):
        for joint_index in range(3):
            rates = leg_motion[:, leg_index * 3 + joint_index, 3]
            angular_speed[f"leg{leg_index + 1}_joint{joint_index + 1}"] = {
                "max_rad_per_frame": float(np.max(np.abs(rates))),
                "rms_rad_per_frame": float(np.sqrt(np.mean(rates ** 2))),
            }

    angles = {}
    for triangle_name, triangle in (
        ("front", config.FRONT_TRIANGLE),
        ("rear", config.REAR_TRIANGLE),
    ):
        a, b, c = triangle
        for vertex, p, q in ((a, b, c), (b, a, c), (c, a, b)):
            values = [
                _angle(points[t, p - 1], points[t, vertex - 1], points[t, q - 1])
                for t in range(len(points))
            ]
            angles[f"{triangle_name}_pt{vertex}"] = _angle_summary(values)
    for leg_index, (anchor, chain) in enumerate(zip(config.LEG_ANCHORS, config.LEG_CHAINS), 1):
        root, knee, tip = chain
        anchor_frames = front_frames if anchor <= 3 else rear_frames
        local_directions = np.stack([
            anchor_frames[t].T @ (points[t, root - 1] - points[t, anchor - 1])
            for t in range(len(points))
        ])
        hip_azimuth = np.unwrap(np.arctan2(
            local_directions[:, 1], local_directions[:, 0]
        ))
        hip_elevation = np.degrees(np.arctan2(
            local_directions[:, 2], np.linalg.norm(local_directions[:, :2], axis=-1)
        ))
        virtual_joint_angles = {
            "azimuth_min_deg_unwrapped": float(np.degrees(np.min(hip_azimuth))),
            "azimuth_max_deg_unwrapped": float(np.degrees(np.max(hip_azimuth))),
            "elevation_min_deg": float(np.min(hip_elevation)),
            "elevation_max_deg": float(np.max(hip_elevation)),
        }
        root_values = [
            _angle(points[t, anchor - 1], points[t, root - 1], points[t, knee - 1])
            for t in range(len(points))
        ]
        knee_values = [
            _angle(points[t, root - 1], points[t, knee - 1], points[t, tip - 1])
            for t in range(len(points))
        ]
        angles[f"leg{leg_index}_root_hinge"] = _angle_summary(root_values)
        angles[f"leg{leg_index}_knee_hinge"] = _angle_summary(knee_values)
        angles[f"leg{leg_index}_virtual_hip_relative_to_pt{anchor}"] = virtual_joint_angles

    report = [
        "# Generated by analyze_motion.py",
        f"SOURCE_DATA_FILE = {data_file.name!r}",
        f"TRANSLATED_BODY_FILE = {body_file.name!r}",
        f"TRANSLATED_LEGS_FILE = {legs_file.name!r}",
        f"VIRTUAL_HIP_ANCHORS = {dict(enumerate(config.LEG_ANCHORS, start=1))!r}",
        f"FRAME_COUNT = {len(points)}",
        "# RMS edge length = sqrt(mean(length**2)); edge MSE is temporal variation from that edge's mean.",
        "EDGE_RMS_LENGTH = " + pformat(edge_rms, sort_dicts=False),
        "EDGE_LENGTH_MSE_FROM_MEAN = " + pformat(edge_mse, sort_dicts=False),
        "# Speeds are radians per frame; triangle rates and all 18 relative joints are included.",
        "MAX_AND_RMS_ANGULAR_SPEED = " + pformat(angular_speed, sort_dicts=False),
        "# Per-frame change in the front triangle's normal direction.",
        "FRONT_TRIANGLE_NORMAL_TURN_LIMITS = " + pformat(front_normal_turn_limits, sort_dicts=False),
        "REAR_FRONT_NORMAL_PARALLEL_LIMITS = " + pformat(rear_front_normal_limits, sort_dicts=False),
        "HEAD_PITCH_RELATIVE_TO_GROUND = " + pformat(head_pitch_summary, sort_dicts=False),
        "# Hinge-angle summaries use empirical p01..p99; virtual-hip azimuth/elevation report observed min/max.",
        "EMPIRICAL_ANGLE_LIMITS_DEG = " + pformat(angles, sort_dicts=False),
        "",
    ]
    config.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = config.ANALYSIS_DIR / f"{name}_motion_analysis.txt"
    report_path.write_text("\n\n".join(report), encoding="utf-8")
    print(f"Analyzed: {data_file.name}, {body_file.name}, {legs_file.name}")
    print(f"Wrote Python-style report: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-file", default=f"{config.TARGET_NAME}_smoothed_coords.csv",
        help="source CSV basename in data/ or an absolute path",
    )
    args = parser.parse_args()
    analyze(Path(args.data_file))


if __name__ == "__main__":
    main()
