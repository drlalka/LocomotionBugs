"""Predict joint velocities from the first N source rows and reconstruct motion."""

from pathlib import Path
import json
import argparse

import numpy as np
import pandas as pd
import torch

from dots_to_rotato import (
    DATA_DIR,
    coordinates_to_features,
    feature_columns,
    load_coordinates,
    to_canonical,
)
from Learn import CHECKPOINT_PATH, DEVICE, JointVelocityGRU
from rotato_to_dots import build_state, integrate, to_source_coordinates


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = DATA_DIR / "Scarites_6_smoothed_coords.csv"
SEED_ROWS = 100
NUM_GENERATED_FRAMES = 1000
VELOCITY_OUTPUT = DATA_DIR / "Scarites_6_generated_velocities.csv"
POINT_OUTPUT = DATA_DIR / "Scarites_6_generated_rotato.csv"


def limit_angular_step(previous, candidate, max_speed, max_acceleration):
    """Apply measured angular-speed and angular-acceleration caps per frame."""
    result = candidate.copy()
    old_joints = previous[3:].reshape(18, 7)
    new_joints = result[3:].reshape(18, 7)
    for index in range(18):
        old_axis, old_rate = old_joints[index, 3:6], old_joints[index, 6]
        old_norm = np.linalg.norm(old_axis)
        old_omega = old_axis / old_norm * old_rate if old_norm > 1e-8 else np.zeros(3)

        axis, rate = new_joints[index, 3:6], float(new_joints[index, 6])
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-8:
            omega = np.zeros(3)
        else:
            omega = axis / axis_norm * np.clip(rate, 0.0, max_speed)

        delta = omega - old_omega
        delta_norm = np.linalg.norm(delta)
        if delta_norm > max_acceleration:
            omega = old_omega + delta * (max_acceleration / delta_norm)
        omega_norm = np.linalg.norm(omega)
        if omega_norm > max_speed:
            omega *= max_speed / omega_norm
            omega_norm = max_speed
        if omega_norm > 1e-8:
            new_joints[index, 3:6] = omega / omega_norm
            new_joints[index, 6] = omega_norm
        else:
            new_joints[index, 3:7] = 0.0
    result[3:] = new_joints.reshape(-1)
    return result


def main(seed_rows=SEED_ROWS):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    source, point_columns = load_coordinates(SOURCE_PATH)
    seed_rows = int(seed_rows)
    history_size = int(checkpoint["n_steps"])
    if seed_rows < history_size:
        raise ValueError(
            f"seed_rows must be at least the model history length ({history_size}); "
            f"received {seed_rows}"
        )
    if len(source) < seed_rows:
        raise ValueError(f"Need {seed_rows} seed rows; found {len(source)}")
    origin = source[0, 0].copy()
    basis = to_canonical(source[:seed_rows])[2]
    canonical, _, _ = to_canonical(source, origin=origin, basis=basis)
    all_features = coordinates_to_features(canonical)
    seed_features = all_features[:seed_rows]
    state = build_state(canonical, seed_rows - 1, origin, basis)
    state["source_file"] = SOURCE_PATH.name
    state["seed_rows"] = seed_rows
    state_output = DATA_DIR / f"Scarites_6_state_row{seed_rows}.json"

    model = JointVelocityGRU().to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["feature_scale"], dtype=np.float32)
    horizon = int(checkpoint["prediction_steps"])
    max_speed = float(checkpoint["max_angular_speed"])
    max_acceleration = float(checkpoint["max_angular_acceleration"])
    history = ((seed_features - mean) / scale).astype(np.float32)[-history_size:]
    generated = []

    with torch.no_grad():
        while len(generated) < NUM_GENERATED_FRAMES:
            x = torch.tensor(history[None], dtype=torch.float32, device=DEVICE)
            pred_scaled = model(x)[0].cpu().numpy()
            pred_raw = pred_scaled * scale + mean
            previous = history[-1] * scale + mean
            limited = []
            for row in pred_raw:
                row = limit_angular_step(previous, row, max_speed, max_acceleration)
                limited.append(row)
                previous = row
            limited = np.asarray(limited, dtype=np.float32)
            count = min(horizon, NUM_GENERATED_FRAMES - len(generated))
            generated.extend(limited[:count])
            normalized = ((limited - mean) / scale).astype(np.float32)
            history = np.concatenate((history, normalized), axis=0)[-history_size:]
            if len(generated) % 100 < horizon or len(generated) == NUM_GENERATED_FRAMES:
                print(f"Predicted {len(generated)}/{NUM_GENERATED_FRAMES} frames")

    generated_features = np.asarray(generated[:NUM_GENERATED_FRAMES], dtype=np.float32)
    pd.DataFrame(generated_features, columns=feature_columns()).to_csv(
        VELOCITY_OUTPUT, index=False, float_format="%.9f"
    )
    state_output.write_text(json.dumps(state, indent=2), encoding="utf-8")

    future_canonical = integrate(state, generated_features)
    future_source = to_source_coordinates(future_canonical, origin, basis)
    pd.DataFrame(future_source.reshape(-1, 69), columns=point_columns).to_csv(
        POINT_OUTPUT, index=False, float_format="%.8f"
    )
    print(f"Used first {seed_rows} source rows; state is at source row {seed_rows}.")
    print(f"Saved {len(generated_features)} generated velocity rows: {VELOCITY_OUTPUT}")
    print(f"Saved state at source row {seed_rows}: {state_output}")
    print(f"Saved {len(future_source)} generated coordinate rows: {POINT_OUTPUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-rows", type=int, default=SEED_ROWS,
                        help="number of initial source CSV rows used as the seed")
    main(parser.parse_args().seed_rows)
