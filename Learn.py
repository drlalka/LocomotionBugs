"""Train one GRU to predict center velocity and all joint angular velocities."""

from pathlib import Path
import copy
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from dots_to_rotato import DATA_DIR, FEATURE_COUNT, feature_columns


ROOT = Path(__file__).resolve().parent
MOTION_DIR = DATA_DIR / "rotato"
CHECKPOINT_PATH = ROOT / "rotato_gru.pt"
N_STEPS = 50
PREDICTION_STEPS = 7
TRAIN_RATIO = 0.8
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.1
EPOCHS = 80
BATCH_SIZE = 64
LEARNING_RATE = 8e-4
PATIENCE = 12
SEED = 42

CENTER_FIT_WEIGHT = 2.0
JOINT_POINT_VELOCITY_FIT_WEIGHT = 1.0
ANGULAR_VELOCITY_FIT_WEIGHT = 20.0
ANGULAR_ACCELERATION_CAP_WEIGHT = 100.0
ANGULAR_SPEED_CAP_WEIGHT = 50.0

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    torch.set_num_threads(2)


def measure_limits(arrays):
    speed_samples, acceleration_samples = [], []
    for features in arrays:
        joints = features[:, 3:].reshape(-1, 18, 7)
        omega = joints[:, :, 3:6] * joints[:, :, 6:7]
        speed_samples.append(np.abs(joints[:, :, 6]).reshape(-1))
        acceleration_samples.append(np.linalg.norm(np.diff(omega, axis=0), axis=-1).reshape(-1))
    max_speed = max(float(np.percentile(x, 99.5)) for x in speed_samples) * 1.2
    max_acceleration = max(float(np.percentile(x, 99.5)) for x in acceleration_samples) * 1.2
    return max(max_speed, 1e-3), max(max_acceleration, 1e-3)


def load_motion_files():
    paths = sorted(MOTION_DIR.glob("*_rotato.csv"))
    if len(paths) < 2:
        raise FileNotFoundError(
            f"Expected motion CSVs in {MOTION_DIR}; run dots_to_rotato.py first"
        )
    arrays = []
    for path in paths:
        frame = pd.read_csv(path)
        if list(frame.columns) != feature_columns():
            raise ValueError(f"Unexpected feature columns in {path.name}")
        values = frame.to_numpy(np.float32)
        if values.shape[1] != FEATURE_COUNT or not np.isfinite(values).all():
            raise ValueError(f"Invalid motion data in {path.name}")
        arrays.append((path.stem.replace("_rotato", ""), values))
    return arrays


def make_windows(arrays, mean, scale):
    train_x, train_y, val_x, val_y = [], [], [], []
    for _, values in arrays:
        normalized = ((values - mean) / scale).astype(np.float32)
        split = int(len(values) * TRAIN_RATIO)
        for start in range(N_STEPS, split - PREDICTION_STEPS + 1):
            train_x.append(normalized[start - N_STEPS:start])
            train_y.append(normalized[start:start + PREDICTION_STEPS])
        for start in range(max(N_STEPS, split), len(values) - PREDICTION_STEPS + 1):
            val_x.append(normalized[start - N_STEPS:start])
            val_y.append(normalized[start:start + PREDICTION_STEPS])

    def pack(xs, ys):
        return (
            torch.tensor(np.asarray(xs), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.asarray(ys), dtype=torch.float32, device=DEVICE),
        )

    return pack(train_x, train_y), pack(val_x, val_y)


class JointVelocityGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=FEATURE_COUNT,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, PREDICTION_STEPS * FEATURE_COUNT),
        )

    def forward(self, x):
        encoded, _ = self.gru(x)
        return self.head(encoded[:, -1]).reshape(-1, PREDICTION_STEPS, FEATURE_COUNT)


def angular_vectors(raw_features):
    joints = raw_features[..., 3:].reshape(*raw_features.shape[:-1], 18, 7)
    return joints[..., 3:6] * joints[..., 6:7]


def loss_fn(pred_scaled, target_scaled, previous_scaled, mean, scale,
            max_speed, max_acceleration):
    pred = pred_scaled * scale + mean
    target = target_scaled * scale + mean
    previous = previous_scaled * scale + mean

    center_fit = F.mse_loss(pred_scaled[..., :3], target_scaled[..., :3])
    pred_joint = pred[..., 3:].reshape(-1, PREDICTION_STEPS, 18, 7)
    target_joint = target[..., 3:].reshape(-1, PREDICTION_STEPS, 18, 7)
    joint_velocity_scale = scale[..., 3:].reshape(18, 7)[..., :3].clamp_min(1e-3)
    joint_point_velocity_fit = F.smooth_l1_loss(
        (pred_joint[..., :3] - target_joint[..., :3]) / joint_velocity_scale,
        torch.zeros_like(pred_joint[..., :3]),
    )
    omega_pred = pred_joint[..., 3:6] * pred_joint[..., 6:7]
    omega_target = target_joint[..., 3:6] * target_joint[..., 6:7]
    omega_fit = ((omega_pred - omega_target) / max_speed).square().mean()
    speed_fit = ((pred_joint[..., 6] - target_joint[..., 6]) / max_speed).square().mean()
    moving_mask = (target_joint[..., 6].abs() > 1e-3).unsqueeze(-1)
    axis_fit = ((pred_joint[..., 3:6] - target_joint[..., 3:6]).square() * moving_mask).sum()
    axis_fit = axis_fit / moving_mask.expand_as(pred_joint[..., 3:6]).sum().clamp_min(1)

    speed_excess = F.relu(pred_joint[..., 6].abs() - max_speed) / max_speed
    speed_cap = speed_excess.square().mean()

    previous_joint = previous[..., 3:].reshape(-1, 18, 7)
    previous_omega = previous_joint[..., 3:6] * previous_joint[..., 6:7]
    omega_sequence = torch.cat((previous_omega.unsqueeze(1), omega_pred), dim=1)
    angular_acceleration = torch.linalg.vector_norm(
        omega_sequence[:, 1:] - omega_sequence[:, :-1], dim=-1
    )
    acceleration_excess = F.relu(angular_acceleration - max_acceleration) / max_acceleration
    acceleration_cap = acceleration_excess.square().mean()

    total = (
        CENTER_FIT_WEIGHT * center_fit
        + JOINT_POINT_VELOCITY_FIT_WEIGHT * joint_point_velocity_fit
        + ANGULAR_VELOCITY_FIT_WEIGHT * (omega_fit + 0.25 * speed_fit + 0.1 * axis_fit)
        + ANGULAR_ACCELERATION_CAP_WEIGHT * acceleration_cap
        + ANGULAR_SPEED_CAP_WEIGHT * speed_cap
    )
    return total, {
        "center_fit": center_fit,
        "joint_point_velocity_fit": joint_point_velocity_fit,
        "angular_velocity_fit": omega_fit,
        "angular_acceleration_cap": acceleration_cap,
        "angular_speed_cap": speed_cap,
    }


def main():
    named_arrays = load_motion_files()
    arrays = [values for _, values in named_arrays]
    split_arrays = [values[:int(len(values) * TRAIN_RATIO)] for values in arrays]
    train_values = np.concatenate(split_arrays, axis=0)
    mean = train_values.mean(axis=0).astype(np.float32)
    scale = train_values.std(axis=0).clip(1e-3).astype(np.float32)
    max_speed, max_acceleration = measure_limits(split_arrays)
    print(f"Motion datasets: {[name for name, _ in named_arrays]}")
    print(f"History N={N_STEPS}; forecast M={PREDICTION_STEPS}; features={FEATURE_COUNT}")
    print(f"Angular speed cap: {max_speed:.4f} rad/frame")
    print(f"Angular acceleration cap: {max_acceleration:.4f} rad/frame²")
    (train_x, train_y), (val_x, val_y) = make_windows(named_arrays, mean, scale)
    print(f"Training windows: {len(train_x)}; validation windows: {len(val_x)}")

    mean_t = torch.tensor(mean, device=DEVICE).view(1, 1, -1)
    scale_t = torch.tensor(scale, device=DEVICE).view(1, 1, -1)
    max_speed_t = torch.tensor(max_speed, device=DEVICE)
    max_acc_t = torch.tensor(max_acceleration, device=DEVICE)
    model = JointVelocityGRU().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_loss, best_state, wait = float("inf"), None, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train_x), device=DEVICE)
        train_total = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            pred = model(train_x[batch])
            loss, _ = loss_fn(
                pred, train_y[batch], train_x[batch, -1],
                mean_t, scale_t, max_speed_t, max_acc_t,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_total += float(loss.detach()) * len(batch)

        model.eval()
        with torch.no_grad():
            pred = model(val_x)
            val_loss, parts = loss_fn(
                pred, val_y, val_x[:, -1], mean_t, scale_t,
                max_speed_t, max_acc_t,
            )
        if float(val_loss) < best_loss:
            best_loss = float(val_loss)
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch {epoch:03d} | train {train_total / len(train_x):.5f} "
                f"| val {float(val_loss):.5f} | "
                f"omega {float(parts['angular_velocity_fit']):.5f} "
                f"accel-cap {float(parts['angular_acceleration_cap']):.5f}"
            )
        if wait >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("Training produced no valid checkpoint")
    model.load_state_dict(best_state)
    torch.save({
        "model_state": model.state_dict(),
        "feature_columns": feature_columns(),
        "feature_mean": mean,
        "feature_scale": scale,
        "n_steps": N_STEPS,
        "prediction_steps": PREDICTION_STEPS,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "max_angular_speed": max_speed,
        "max_angular_acceleration": max_acceleration,
        "loss_weights": {
            "center_fit": CENTER_FIT_WEIGHT,
            "joint_point_velocity_fit": JOINT_POINT_VELOCITY_FIT_WEIGHT,
            "angular_velocity_fit": ANGULAR_VELOCITY_FIT_WEIGHT,
            "angular_acceleration_cap": ANGULAR_ACCELERATION_CAP_WEIGHT,
            "angular_speed_cap": ANGULAR_SPEED_CAP_WEIGHT,
        },
        "datasets": [name for name, _ in named_arrays],
    }, CHECKPOINT_PATH)
    print(f"Saved one joint-velocity model: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
