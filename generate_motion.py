"""Stage 2: autoregressively generate body and leg angular velocities."""

from datetime import datetime
import json

import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation

import config
from Learn import DEVICE, MotionGRU
from dots_to_rotato import BODY_COLUMNS, LEG_COLUMNS
from rotato_to_dots import _initial_state_from_source


def _limited_angular(previous, predicted, cap_speed, cap_accel):
    old_axis = previous[:3]
    old_rate = float(previous[3])
    old_norm = np.linalg.norm(old_axis)
    old_omega = old_axis / old_norm * old_rate if old_norm > 1e-9 else np.zeros(3)
    axis = predicted[:3]
    norm = np.linalg.norm(axis)
    omega = axis / norm * np.clip(float(predicted[3]), -cap_speed, cap_speed) if norm > 1e-9 else np.zeros(3)
    delta = omega - old_omega
    delta_norm = np.linalg.norm(delta)
    if delta_norm > cap_accel:
        omega = old_omega + delta * (cap_accel / delta_norm)
    speed = np.linalg.norm(omega)
    if speed > cap_speed:
        omega *= cap_speed / speed
        speed = cap_speed
    if speed < 1e-9:
        return np.zeros(4, dtype=np.float32)
    return np.concatenate((omega / speed, [speed])).astype(np.float32)


def _limit_row(previous_body, body, previous_legs, legs, caps):
    body = body.copy()
    legs = legs.copy().reshape(config.JOINT_COUNT, 4)
    body_speed_caps = caps.get("body_max_angular_speed", [caps["max_angular_speed"]] * 2)
    for joint, (axis_slice, rate_index) in enumerate(((slice(3, 6), 6), (slice(7, 10), 10))):
        prev = np.concatenate((previous_body[axis_slice], [previous_body[rate_index]]))
        pred = np.concatenate((body[axis_slice], [body[rate_index]]))
        limited = _limited_angular(prev, pred, body_speed_caps[joint], caps["max_angular_acceleration"])
        body[axis_slice] = limited[:3]
        body[rate_index] = limited[3]
    previous_legs = previous_legs.reshape(config.JOINT_COUNT, 4)
    leg_speed_caps = caps.get(
        "leg_max_angular_speed", [caps["max_angular_speed"]] * config.JOINT_COUNT
    )
    for j in range(config.JOINT_COUNT):
        legs[j] = _limited_angular(
            previous_legs[j], legs[j], leg_speed_caps[j],
            caps["max_angular_acceleration"],
        )
    return body, legs.reshape(-1)


def _steer_front_to_velocity(body_row, previous_body, front_rotation,
                             head_local, up, caps):
    """Apply a capped yaw correction so the head points along COM travel."""
    body_row = body_row.copy()
    old_rotation = front_rotation
    axis = body_row[3:6]
    axis_norm = np.linalg.norm(axis)
    predicted_delta = (
        Rotation.from_rotvec(axis / axis_norm * float(body_row[6])).as_matrix()
        if axis_norm > 1e-9 and abs(float(body_row[6])) > 1e-10 else np.eye(3)
    )
    candidate = predicted_delta @ old_rotation
    velocity = body_row[:3].copy()
    velocity -= np.dot(velocity, up) * up
    heading = candidate @ head_local
    heading -= np.dot(heading, up) * up
    if np.linalg.norm(velocity) < config.HEADING_MIN_SPEED or np.linalg.norm(heading) < 1e-9:
        return body_row, candidate

    velocity /= np.linalg.norm(velocity)
    heading /= np.linalg.norm(heading)
    yaw = np.arctan2(np.dot(up, np.cross(heading, velocity)), np.dot(heading, velocity))
    corrected = Rotation.from_rotvec(up * yaw).as_matrix() @ candidate
    delta = corrected @ old_rotation.T
    rotvec = Rotation.from_matrix(delta).as_rotvec()
    rate = float(np.linalg.norm(rotvec))
    requested = np.zeros(4, dtype=np.float32)
    if rate > 1e-9:
        requested[:3] = rotvec / rate
        requested[3] = rate
    previous = np.concatenate((previous_body[3:6], [previous_body[6]]))
    front_speed_cap = caps.get("body_max_angular_speed", [caps["max_angular_speed"]])[0]
    limited = _limited_angular(
        previous, requested, front_speed_cap, caps["max_angular_acceleration"]
    )
    body_row[3:6] = limited[:3]
    body_row[6] = limited[3]
    if limited[3] > 1e-9:
        front_rotation = Rotation.from_rotvec(limited[:3] * limited[3]).as_matrix() @ old_rotation
    else:
        front_rotation = old_rotation
    return body_row, front_rotation


def _next_run_name():
    stamp = datetime.now().strftime("%H%M")
    base = f"{config.TARGET_NAME}_{config.RUN_PARAMETER_TAG}_T{stamp}"
    name, suffix = base, 1
    while (config.GENERATION_DIR / f"{name}_body.csv").exists():
        name = f"{base}_{suffix:02d}"
        suffix += 1
    return name


def main():
    body_path = config.TRANSLATE_DIR / f"{config.TARGET_NAME}_body.csv"
    legs_path = config.TRANSLATE_DIR / f"{config.TARGET_NAME}_legs.csv"
    body_seed = pd.read_csv(body_path)[BODY_COLUMNS].to_numpy(np.float32)
    leg_seed = pd.read_csv(legs_path)[LEG_COLUMNS].to_numpy(np.float32)
    if len(body_seed) != len(leg_seed):
        raise ValueError("Translated body and leg histories have different lengths")
    if len(body_seed) < config.HISTORY_N:
        raise ValueError(f"Need at least {config.HISTORY_N} translated seed rows")

    body_checkpoint = torch.load(config.BODY_MODEL_PATH, map_location=DEVICE, weights_only=False)
    legs_checkpoint = torch.load(config.LEGS_MODEL_PATH, map_location=DEVICE, weights_only=False)
    body_model = MotionGRU(config.BODY_FEATURES).to(DEVICE)
    legs_model = MotionGRU(config.LEG_FEATURES).to(DEVICE)
    body_model.load_state_dict(body_checkpoint["model_state"])
    legs_model.load_state_dict(legs_checkpoint["model_state"])
    body_model.eval()
    legs_model.eval()

    x_mean = np.asarray(body_checkpoint["input_mean"], dtype=np.float32)
    x_scale = np.asarray(body_checkpoint["input_scale"], dtype=np.float32)
    body_mean = np.asarray(body_checkpoint["output_mean"], dtype=np.float32)
    body_scale = np.asarray(body_checkpoint["output_scale"], dtype=np.float32)
    legs_mean = np.asarray(legs_checkpoint["output_mean"], dtype=np.float32)
    legs_scale = np.asarray(legs_checkpoint["output_scale"], dtype=np.float32)
    caps = body_checkpoint["caps"]
    full_seed = np.concatenate((body_seed, leg_seed), axis=1)
    history = ((full_seed[:config.HISTORY_N] - x_mean) / x_scale).astype(np.float32)

    generated_body, generated_legs = [], []
    previous_body = body_seed[config.HISTORY_N - 1].copy()
    previous_legs = leg_seed[config.HISTORY_N - 1].copy()
    seed_source = config.DATA_DIR / f"{config.TARGET_NAME}_smoothed_coords.csv"
    seed_state = _initial_state_from_source(seed_source, config.HISTORY_N - 1)
    front_rotation = seed_state["front_rotation"].copy()
    head_local = 0.5 * (seed_state["front_local"][1] + seed_state["front_local"][2]) - seed_state["front_local"][0]
    up = seed_state["basis"][2].copy()
    with torch.no_grad():
        while len(generated_body) < config.GENERATED_STEPS:
            x = torch.tensor(history[None], dtype=torch.float32, device=DEVICE)
            body_scaled = body_model(x)[0].cpu().numpy()
            legs_scaled = legs_model(x)[0].cpu().numpy()
            body_rows = body_scaled * body_scale + body_mean
            leg_rows = legs_scaled * legs_scale + legs_mean
            limited_body, limited_legs = [], []
            prev_b, prev_l = previous_body, previous_legs
            for b_row, l_row in zip(body_rows, leg_rows):
                b_row, l_row = _limit_row(prev_b, b_row, prev_l, l_row, caps)
                b_row, front_rotation = _steer_front_to_velocity(
                    b_row, prev_b, front_rotation, head_local, up, caps
                )
                limited_body.append(b_row)
                limited_legs.append(l_row)
                prev_b, prev_l = b_row, l_row
            limited_body = np.asarray(limited_body, dtype=np.float32)
            limited_legs = np.asarray(limited_legs, dtype=np.float32)
            count = min(config.PREDICTION_M, config.GENERATED_STEPS - len(generated_body))
            generated_body.extend(limited_body[:count])
            generated_legs.extend(limited_legs[:count])
            combined = np.concatenate((limited_body, limited_legs), axis=1)
            history = np.concatenate((history, (combined - x_mean) / x_scale), axis=0)[-config.HISTORY_N:]
            previous_body, previous_legs = limited_body[-1], limited_legs[-1]
            if len(generated_body) % 100 < config.PREDICTION_M or len(generated_body) == config.GENERATED_STEPS:
                print(f"generated velocities {len(generated_body)}/{config.GENERATED_STEPS}")

    body_out = np.asarray(generated_body[:config.GENERATED_STEPS], dtype=np.float32)
    legs_out = np.asarray(generated_legs[:config.GENERATED_STEPS], dtype=np.float32)
    config.GENERATION_DIR.mkdir(parents=True, exist_ok=True)
    run_name = _next_run_name()
    body_out_path = config.GENERATION_DIR / f"{run_name}_body.csv"
    legs_out_path = config.GENERATION_DIR / f"{run_name}_legs.csv"
    pd.DataFrame(body_out, columns=BODY_COLUMNS).to_csv(body_out_path, index=False, float_format="%.9f")
    pd.DataFrame(legs_out, columns=LEG_COLUMNS).to_csv(legs_out_path, index=False, float_format="%.9f")
    manifest = {
        "run_name": run_name,
        "target": config.TARGET_NAME,
        "seed_dataset": f"{config.TARGET_NAME}_smoothed_coords.csv",
        "seed_frame": config.HISTORY_N - 1,
        "train_prefix": config.TRAIN_PREFIX,
        "training_datasets": body_checkpoint["datasets"],
        "history_n": config.HISTORY_N,
        "prediction_m": config.PREDICTION_M,
        "generated_steps": config.GENERATED_STEPS,
        "body_file": body_out_path.name,
        "legs_file": legs_out_path.name,
        "caps": caps,
    }
    (config.GENERATION_DIR / "latest_run.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"generated velocity run: {run_name}")


if __name__ == "__main__":
    main()
