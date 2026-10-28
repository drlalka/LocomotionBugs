"""Stage 1: train body-motion and all-leg angular-velocity GRUs."""

import copy
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

import config
from dots_to_rotato import (
    BODY_COLUMNS, LEG_COLUMNS, canonicalize, load_coordinates, triangle_frame,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
random.seed(config.RANDOM_SEED)
if DEVICE.type == "cpu":
    torch.set_num_threads(2)

FRONT_FRAMES_BY_NAME = {}


class MotionGRU(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.gru = nn.GRU(
            config.INPUT_FEATURES,
            config.HIDDEN_SIZE,
            num_layers=config.GRU_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.GRU_LAYERS > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_SIZE, config.PREDICTION_M * output_size),
        )
        self.output_size = output_size

    def forward(self, x):
        encoded, _ = self.gru(x)
        return self.head(encoded[:, -1]).reshape(-1, config.PREDICTION_M, self.output_size)


def _load_motion_pairs():
    paths = config.selected_training_paths()
    pairs = []
    for source_path in paths:
        name = source_path.stem.replace("_smoothed_coords", "")
        body_path = config.TRANSLATE_DIR / f"{name}_body.csv"
        leg_path = config.TRANSLATE_DIR / f"{name}_legs.csv"
        if not body_path.exists() or not leg_path.exists():
            raise FileNotFoundError(
                f"Missing translated features for selected dataset {source_path.name}; run stage 0"
            )
        body = pd.read_csv(body_path)[BODY_COLUMNS].to_numpy(np.float32)
        legs = pd.read_csv(leg_path)[LEG_COLUMNS].to_numpy(np.float32)
        if len(body) != len(legs):
            raise ValueError(f"Frame count mismatch for {name}")
        points, _ = load_coordinates(source_path)
        canonical, _, _ = canonicalize(points)
        front_points = canonical[:, np.asarray(config.FRONT_TRIANGLE) - 1]
        front_frames = np.stack([triangle_frame(row) for row in front_points])
        if len(front_frames) != len(body):
            raise ValueError(f"Pose orientation count mismatch for {name}")
        FRONT_FRAMES_BY_NAME[name] = front_frames.astype(np.float32)
        pairs.append((name, body, legs))
    if not pairs:
        raise FileNotFoundError("No selected translated datasets found; run stage 0 first")
    return pairs


def _measure_caps(pairs):
    acceleration_quantiles = []
    training_leg_rates = []
    recorded_leg_rates, recorded_body_rates = [], []
    for _, body, legs in pairs:
        recorded_leg_rates.append(np.abs(legs.reshape(-1, config.JOINT_COUNT, 4)[..., 3]))
        recorded_body_rates.append(np.abs(body[:, [6, 10]]))
        training_leg_rates.append(np.abs(
            legs[:int(len(legs) * config.TRAIN_FRACTION)]
            .reshape(-1, config.JOINT_COUNT, 4)[..., 3]
        ))
        body_angular = np.stack((body[:, 3:7], body[:, 7:11]), axis=1)
        leg_angular = legs.reshape(len(legs), config.JOINT_COUNT, 4)
        angular = np.concatenate((body_angular, leg_angular), axis=1)
        omega = angular[..., :3] * angular[..., 3:4]
        acceleration = np.linalg.norm(np.diff(omega, axis=0), axis=-1)
        acceleration_quantiles.append(float(np.percentile(acceleration, 99.5)))
    leg_rates = np.concatenate(training_leg_rates, axis=0)
    # Actual maxima per joint, including recorded pauses and outliers. No
    # percentile/margin or arbitrary fallback replaces these measured maxima.
    leg_max = np.concatenate(recorded_leg_rates, axis=0).max(axis=0)
    body_max = np.concatenate(recorded_body_rates, axis=0).max(axis=0)
    leg_min = []
    for joint in range(config.JOINT_COUNT):
        active = leg_rates[:, joint][leg_rates[:, joint] > 1e-5]
        floor = (np.percentile(active, config.LEG_MIN_SPEED_PERCENTILE)
                 * config.LEG_MIN_SPEED_FRACTION) if len(active) else 0.0
        leg_min.append(float(min(floor, leg_max[joint] * 0.5)))
    return {
        "leg_rms_angular_speed": np.sqrt(np.mean(leg_rates ** 2, axis=0)).clip(1e-4).tolist(),
        "leg_min_angular_speed": leg_min,
        "leg_max_angular_speed": leg_max.tolist(),
        "body_max_angular_speed": body_max.tolist(),
        "max_angular_speed": float(max(leg_max.max(), body_max.max())),
        "max_angular_acceleration": max(
            config.DEFAULT_MAX_ANGULAR_ACCELERATION,
            max(acceleration_quantiles) * config.ANGULAR_ACCELERATION_MARGIN,
        ),
    }


def _make_windows(pairs, x_mean, x_scale, y_mean, y_scale, head):
    inputs_train, outputs_train, poses_train = [], [], []
    inputs_val, outputs_val, poses_val = [], [], []
    for name, body, legs in pairs:
        front_frames = FRONT_FRAMES_BY_NAME[name]
        full = np.concatenate((body, legs), axis=1)
        target = body if head == "body" else legs
        x = ((full - x_mean) / x_scale).astype(np.float32)
        y = ((target - y_mean) / y_scale).astype(np.float32)
        split = int(len(full) * config.TRAIN_FRACTION)
        for start in range(config.HISTORY_N, split - config.PREDICTION_M + 1):
            inputs_train.append(x[start - config.HISTORY_N:start])
            outputs_train.append(y[start:start + config.PREDICTION_M])
            poses_train.append(front_frames[start - 1])
        for start in range(max(config.HISTORY_N, split), len(full) - config.PREDICTION_M + 1):
            inputs_val.append(x[start - config.HISTORY_N:start])
            outputs_val.append(y[start:start + config.PREDICTION_M])
            poses_val.append(front_frames[start - 1])

    def pack(xs, ys, poses):
        return (
            torch.tensor(np.asarray(xs), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.asarray(ys), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.asarray(poses), dtype=torch.float32, device=DEVICE),
        )

    return (
        pack(inputs_train, outputs_train, poses_train),
        pack(inputs_val, outputs_val, poses_val),
    )


def _angular_view(raw, head):
    if head == "body":
        return torch.stack((raw[..., 3:7], raw[..., 7:11]), dim=-2)
    return raw.reshape(*raw.shape[:-1], config.JOINT_COUNT, 4)


def _rotation_delta(axis_rate):
    """Differentiable Rodrigues rotation from axis * angular-rate (rad/frame)."""
    axis = axis_rate[..., :3]
    axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1e-7)
    vector = axis * axis_rate[..., 3:4]
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((
        zero, -z, y, z, zero, -x, -y, x, zero,
    ), dim=-1).reshape(*vector.shape[:-1], 3, 3)
    theta = torch.linalg.vector_norm(vector, dim=-1, keepdim=True).unsqueeze(-1)
    theta_sq = theta.square()
    a = torch.sin(theta) / theta.clamp_min(1e-7)
    b = (1.0 - torch.cos(theta)) / theta_sq.clamp_min(1e-7)
    eye = torch.eye(3, dtype=vector.dtype, device=vector.device)
    return eye + a * skew + b * (skew @ skew)


def _exponential_speed_penalty(rates, recorded_max):
    """Zero inside the recorded range; exponential growth above its maximum."""
    excess = F.relu(rates.abs() / recorded_max.clamp_min(1e-5) - 1.0)
    exponent = config.ANGULAR_SPEED_EXP_STEEPNESS * excess
    # Linear continuation above exp(20) prevents overflow without losing the
    # restoring gradient for exceptionally bad predictions.
    return torch.expm1(exponent.clamp(max=20.0)) + np.exp(20.0) * F.relu(exponent - 20.0)


def _leg_ground_direction_penalty(direction, up):
    """Signed tilt of a proximal-to-distal link relative to the actual ground."""
    vertical = torch.sum(direction * up, dim=-1)
    minimum_down = np.sin(np.deg2rad(config.LEG_GROUND_MIN_DOWNWARD_DEG))
    # A magnitude-only parallelism penalty would also accept upside-down legs.
    return (F.relu(vertical + minimum_down) / max(minimum_down, 1e-3)).square()


def _geometry_penalties(pred, previous_raw, head, prior, initial_front_rotation=None):
    """Forward-kinematic height penalties for abdomen and distal leg points."""
    if prior is None:
        zero = pred.new_zeros(())
        return zero, zero, zero, zero, zero, zero, zero
    device, dtype = pred.device, pred.dtype
    eye = torch.eye(3, dtype=dtype, device=device)
    front_local = torch.as_tensor(prior["front_local_points"], dtype=dtype, device=device)
    rear_local = torch.as_tensor(prior["rear_local_points_from_pt1"], dtype=dtype, device=device)
    rear_initial = torch.as_tensor(prior["rear_initial_relative_rotation"], dtype=dtype, device=device)
    leg_initial = torch.as_tensor(prior["leg_initial_relative_rotations"], dtype=dtype, device=device).reshape(6, 3, 3, 3)
    lengths = torch.as_tensor(prior["leg_lengths"], dtype=dtype, device=device)
    up = torch.as_tensor(prior["canonical_up"], dtype=dtype, device=device)
    min_belly = float(prior["belly_min_height"])
    max_foot = float(prior["foot_max_height"])
    max_normal_turn = float(prior["max_front_normal_turn_rad_per_frame"])
    max_rear_normal_deviation = min(
        float(prior["max_rear_front_normal_deviation_rad"]),
        np.deg2rad(config.REAR_NORMAL_PARALLEL_MAX_DEG),
    )
    scale = max(float(np.median(np.asarray(prior["leg_lengths"]))), 1.0)
    batch = pred.shape[0]
    front_rotation = (
        eye.expand(batch, -1, -1)
        if initial_front_rotation is None else initial_front_rotation
    )
    rear_relative = rear_initial.expand(batch, -1, -1)
    leg_relative = [
        [leg_initial[leg, joint].expand(batch, -1, -1).clone() for joint in range(3)]
        for leg in range(6)
    ]
    angular_pred = _angular_view(pred, head)
    previous_body = previous_raw[..., :config.BODY_FEATURES]
    belly_terms, foot_terms = [], []
    normal_turn_terms, rear_normal_terms, hip_angle_terms = [], [], []
    hip_elevation_terms = []
    ground_direction_terms = []
    hip_limits = prior["virtual_joint_angle_limits"]

    for t in range(config.PREDICTION_M):
        body_step = angular_pred[:, t] if head == "body" else None
        if body_step is None:
            body_step = torch.stack((
                previous_body[:, 3:7], previous_body[:, 7:11]
            ), dim=1)
        previous_normal = front_rotation[:, :, 2]
        front_rotation = _rotation_delta(body_step[:, 0]) @ front_rotation
        rear_relative = _rotation_delta(body_step[:, 1]) @ rear_relative
        rear_rotation = front_rotation @ rear_relative
        current_normal = front_rotation[:, :, 2]
        normal_turn = torch.acos(torch.sum(previous_normal * current_normal, dim=-1).clamp(-1.0, 1.0))
        normal_turn_terms.append(
            (F.relu(normal_turn - max_normal_turn) / max(max_normal_turn, 1e-4)).square()
        )
        front_normal = front_rotation[:, :, 2]
        rear_normal = rear_rotation[:, :, 2]
        normal_dot = torch.sum(front_normal * rear_normal, dim=-1).abs().clamp(0.0, 1.0 - 1e-7)
        rear_normal_deviation = torch.acos(normal_dot)
        rear_normal_terms.append(
            (F.relu(rear_normal_deviation - max_rear_normal_deviation)
             / max(max_rear_normal_deviation, np.deg2rad(1.0))).square()
        )
        pt1_position = (front_rotation @ front_local[0].view(1, 3, 1)).squeeze(-1)

        if head == "body":
            belly_points = pt1_position[:, None, :] + torch.einsum(
                "bij,pj->bpi", rear_rotation, rear_local[1:]
            )
            belly_height = torch.einsum("bpi,i->bp", belly_points, up)
            belly_terms.append(F.relu(min_belly - belly_height).div(scale).square())
        else:
            leg_step = angular_pred[:, t].reshape(batch, 6, 3, 4)
            tip_heights = []
            for leg_index in range(6):
                anchor_point = config.LEG_ANCHORS[leg_index]
                if anchor_point <= 3:
                    parent = front_rotation
                    anchor_local = front_local[anchor_point - 1]
                    anchor = (front_rotation @ anchor_local.view(1, 3, 1)).squeeze(-1)
                else:
                    parent = rear_rotation
                    anchor_local = rear_local[anchor_point - 3]
                    anchor = pt1_position + (
                        rear_rotation @ anchor_local.view(1, 3, 1)
                    ).squeeze(-1)
                tip = anchor
                for joint_index in range(3):
                    leg_relative[leg_index][joint_index] = (
                        _rotation_delta(leg_step[:, leg_index, joint_index])
                        @ leg_relative[leg_index][joint_index]
                    )
                    child = parent @ leg_relative[leg_index][joint_index]
                    link = child[:, :, 2] * lengths[leg_index, joint_index]
                    if joint_index < 2:
                        ground_direction_terms.append(
                            _leg_ground_direction_penalty(child[:, :, 2], up)
                        )
                    tip = tip + link
                    parent = child
                direction = leg_relative[leg_index][0][:, :, 2]
                azimuth = torch.atan2(direction[:, 1], direction[:, 0])
                elevation = torch.atan2(
                    direction[:, 2], torch.linalg.vector_norm(direction[:, :2], dim=-1)
                )
                limits = hip_limits[leg_index]
                az_min, az_max = float(limits["azimuth_min_rad"]), float(limits["azimuth_max_rad"])
                az_center = 0.5 * (az_min + az_max)
                azimuth = az_center + torch.atan2(
                    torch.sin(azimuth - az_center), torch.cos(azimuth - az_center)
                )
                el_min, el_max = float(limits["elevation_min_rad"]), float(limits["elevation_max_rad"])
                az_scale = max(az_max - az_min, np.deg2rad(5.0))
                el_scale = max(el_max - el_min, np.deg2rad(5.0))
                az_excess = F.relu(az_min - azimuth) + F.relu(azimuth - az_max)
                el_excess = F.relu(el_min - elevation) + F.relu(elevation - el_max)
                hip_angle_terms.append(
                    (az_excess / az_scale).square()
                )
                hip_elevation_terms.append((el_excess / el_scale).square())
                tip_heights.append(torch.einsum("bi,i->b", tip, up))
            foot_height = torch.stack(tip_heights, dim=1)
            foot_terms.append(F.relu(foot_height - max_foot).div(scale).square())

    belly_loss = torch.cat([v.reshape(-1) for v in belly_terms]).mean() if belly_terms else pred.new_zeros(())
    foot_loss = torch.cat([v.reshape(-1) for v in foot_terms]).mean() if foot_terms else pred.new_zeros(())
    normal_loss = torch.cat([v.reshape(-1) for v in normal_turn_terms]).mean() if normal_turn_terms else pred.new_zeros(())
    rear_normal_loss = torch.cat([v.reshape(-1) for v in rear_normal_terms]).mean() if rear_normal_terms else pred.new_zeros(())
    hip_angle_loss = torch.cat([v.reshape(-1) for v in hip_angle_terms]).mean() if hip_angle_terms else pred.new_zeros(())
    hip_elevation_loss = torch.cat([v.reshape(-1) for v in hip_elevation_terms]).mean() if hip_elevation_terms else pred.new_zeros(())
    ground_direction_loss = torch.cat([v.reshape(-1) for v in ground_direction_terms]).mean() if ground_direction_terms else pred.new_zeros(())
    return belly_loss, foot_loss, normal_loss, hip_angle_loss, rear_normal_loss, hip_elevation_loss, ground_direction_loss


def _heading_alignment_losses(pred, initial_front_rotation, prior):
    """Align tracked head direction with travel and keep it level with the ground."""
    if initial_front_rotation is None or prior is None:
        zero = pred.new_zeros(())
        return zero, zero, zero
    front_local = torch.as_tensor(
        prior["front_local_points"], dtype=pred.dtype, device=pred.device
    )
    head_local = 0.5 * (front_local[1] + front_local[2]) - front_local[0]
    up = torch.as_tensor(prior["canonical_up"], dtype=pred.dtype, device=pred.device)
    max_vertical = float(np.sin(prior["max_head_tilt_rad"]))
    heading_rotation = initial_front_rotation
    max_turn = float(prior["max_angular_speed"])
    alignment_terms, level_terms, turn_terms, active_terms = [], [], [], []

    def signed_yaw_error(direction, travel):
        sine = torch.sum(torch.cross(direction, travel, dim=-1) * up, dim=-1)
        cosine = torch.sum(direction * travel, dim=-1)
        return torch.atan2(sine, cosine), cosine

    for t in range(config.PREDICTION_M):
        heading_before = (heading_rotation @ head_local.view(1, 3, 1)).squeeze(-1)
        heading_before_ground = heading_before - torch.sum(
            heading_before * up, dim=-1, keepdim=True
        ) * up
        velocity = pred[:, t, :3]
        velocity_ground = velocity - torch.sum(velocity * up, dim=-1, keepdim=True) * up
        moving = torch.linalg.vector_norm(velocity_ground, dim=-1) > config.HEADING_MIN_SPEED
        yaw_before, cosine_before = signed_yaw_error(
            F.normalize(heading_before_ground, dim=-1),
            F.normalize(velocity_ground, dim=-1),
        )

        heading_rotation = _rotation_delta(_angular_view(pred, "body")[:, t, 0]) @ heading_rotation
        heading = (heading_rotation @ head_local.view(1, 3, 1)).squeeze(-1)
        heading_norm = torch.linalg.vector_norm(heading, dim=-1).clamp_min(1e-7)
        vertical_fraction = torch.abs(torch.sum(heading * up, dim=-1)) / heading_norm
        heading_ground = heading - torch.sum(heading * up, dim=-1, keepdim=True) * up
        yaw_after, cosine_after = signed_yaw_error(
            F.normalize(heading_ground, dim=-1),
            F.normalize(velocity_ground, dim=-1),
        )
        cosine_after = cosine_after.clamp(-1.0, 1.0)
        # Penalize direction error and make facing backward especially costly.
        error = (1.0 - cosine_after).square() + 2.0 * F.relu(-cosine_after).square()
        level_excess = F.relu(vertical_fraction - max_vertical)
        level_scale = max(1.0 - max_vertical, 0.1)
        yaw_error = yaw_before.abs()
        required_turn = torch.clamp(yaw_error, max=max_turn)
        progress = yaw_error - yaw_after.abs()
        shortfall = F.relu(required_turn - progress) / required_turn.clamp_min(np.deg2rad(1.0))
        pitch_before = torch.asin(torch.clamp(
            torch.abs(torch.sum(heading_before * up, dim=-1))
            / torch.linalg.vector_norm(heading_before, dim=-1).clamp_min(1e-7),
            0.0, 1.0,
        ))
        pitch_severity = pitch_before / max(float(prior["max_head_tilt_rad"]), np.deg2rad(1.0))
        turn_weight = 1.0 + pitch_severity + 2.0 * F.relu(-cosine_before)
        needs_turn = moving & (required_turn > np.deg2rad(5.0))
        alignment_terms.append(error * moving.to(error.dtype))
        level_terms.append((level_excess / level_scale).square() * moving.to(error.dtype))
        turn_terms.append(turn_weight * shortfall.square() * needs_turn.to(error.dtype))
        active_terms.append(moving.to(error.dtype))
    valid = torch.stack(active_terms, dim=1).sum().clamp_min(1.0)
    alignment_loss = torch.stack(alignment_terms, dim=1).sum() / valid
    level_loss = torch.stack(level_terms, dim=1).sum() / valid
    turn_loss = torch.stack(turn_terms, dim=1).sum() / valid
    return alignment_loss, level_loss, turn_loss


def _loss(pred_scaled, target_scaled, previous_scaled, input_mean, input_scale,
          output_mean, output_scale, caps, head, geometry_prior=None,
          initial_front_rotation=None):
    pred = pred_scaled * output_scale + output_mean
    target = target_scaled * output_scale + output_mean
    previous_raw = (
        previous_scaled * input_scale.reshape(-1) + input_mean.reshape(-1)
    )
    previous = previous_raw[..., :config.BODY_FEATURES] if head == "body" else previous_raw[..., config.BODY_FEATURES:]
    angular_pred = _angular_view(pred, head)
    angular_target = _angular_view(target, head)
    omega_pred = angular_pred[..., :3] * angular_pred[..., 3:4]
    omega_target = angular_target[..., :3] * angular_target[..., 3:4]
    speed_cap = torch.as_tensor(caps["max_angular_speed"], device=pred.device)
    accel_cap = torch.as_tensor(caps["max_angular_acceleration"], device=pred.device)
    omega_fit = ((omega_pred - omega_target) / speed_cap).square().mean()
    rate_fit = ((angular_pred[..., 3] - angular_target[..., 3]) / speed_cap).square().mean()
    moving = (angular_target[..., 3].abs() > 1e-3).unsqueeze(-1)
    axis_fit = ((angular_pred[..., :3] - angular_target[..., :3]).square() * moving).sum()
    axis_fit = axis_fit / moving.expand_as(angular_pred[..., :3]).sum().clamp_min(1)
    body_upper = pred.new_tensor(caps.get("body_max_angular_speed", [float(speed_cap)] * 2))
    speed_loss = (_exponential_speed_penalty(angular_pred[..., 3], body_upper).mean()
                  if head == "body" else pred.new_zeros(()))
    slow_leg_loss = pred.new_zeros(())
    if head == "legs":
        # Rates are rad/frame; a pause at a step reversal is allowed, while a
        # whole prediction window with an almost motionless joint is penalized.
        rates = angular_pred[..., 3].abs()
        lower = pred.new_tensor(caps["leg_min_angular_speed"])
        upper = pred.new_tensor(caps["leg_max_angular_speed"])
        # Use physical angular-velocity vectors: axis magnitude must not be
        # able to compensate for an incorrect rotation rate.
        omega_pred = F.normalize(angular_pred[..., :3], dim=-1, eps=1e-7) * angular_pred[..., 3:4]
        omega_target = F.normalize(angular_target[..., :3], dim=-1, eps=1e-7) * angular_target[..., 3:4]
        fit_scale = pred.new_tensor(caps.get("leg_rms_angular_speed", caps["leg_max_angular_speed"])).clamp_min(1e-4)
        omega_fit = ((omega_pred - omega_target) / fit_scale[:, None]).square().sum(dim=-1).mean()
        rate_fit = ((angular_pred[..., 3].abs() - angular_target[..., 3].abs()) / fit_scale).square().mean()
        mean_speed = rates.mean(dim=1)
        slow_leg_loss = (F.relu(lower - mean_speed) / lower.clamp_min(1e-5)).square().mean()
        speed_loss = _exponential_speed_penalty(rates, upper).mean()

    previous_angular = _angular_view(previous, head)
    previous_axis = previous_angular[..., :3]
    if head == "legs":
        previous_axis = F.normalize(previous_axis, dim=-1, eps=1e-7)
    previous_omega = previous_axis * previous_angular[..., 3:4]
    omega_sequence = torch.cat((previous_omega.unsqueeze(1), omega_pred), dim=1)
    angular_acceleration = torch.linalg.vector_norm(
        omega_sequence[:, 1:] - omega_sequence[:, :-1], dim=-1
    )
    accel_excess = F.relu(angular_acceleration - accel_cap) / accel_cap
    accel_loss = accel_excess.square().mean()
    belly_low_loss, feet_high_loss, normal_turn_loss, hip_angle_loss, rear_normal_loss, hip_elevation_loss, ground_direction_loss = _geometry_penalties(
        pred, previous_raw, head, geometry_prior, initial_front_rotation
    )
    if head == "body":
        heading_alignment_loss, head_ground_parallel_loss, heading_turn_progress_loss = _heading_alignment_losses(
            pred, initial_front_rotation, geometry_prior
        )
    else:
        heading_alignment_loss = pred.new_zeros(())
        head_ground_parallel_loss = pred.new_zeros(())
        heading_turn_progress_loss = pred.new_zeros(())

    regularization = pred.new_zeros(())
    if head == "body":
        center_loss = F.mse_loss(pred_scaled[..., :3], target_scaled[..., :3])
        total = (
            config.BODY_CENTER_FIT_WEIGHT * center_loss
            + config.BODY_ANGULAR_FIT_WEIGHT * (omega_fit + 0.25 * rate_fit + 0.1 * axis_fit)
            + config.ANGULAR_ACCELERATION_CAP_WEIGHT * accel_loss
            + config.ANGULAR_SPEED_CAP_WEIGHT * speed_loss
            + config.BELLY_TOO_LOW_WEIGHT * belly_low_loss
            + config.FRONT_NORMAL_TURN_WEIGHT * normal_turn_loss
            + config.REAR_NORMAL_PARALLEL_WEIGHT * rear_normal_loss
            + config.HEADING_ALIGNMENT_WEIGHT * heading_alignment_loss
            + config.HEAD_GROUND_PARALLEL_WEIGHT * head_ground_parallel_loss
            + config.HEADING_TURN_PROGRESS_WEIGHT * heading_turn_progress_loss
        )
    else:
        center_loss = pred.new_zeros(())
        auxiliary = (
            (config.ANGULAR_ACCELERATION_CAP_WEIGHT, accel_loss),
            (config.LEG_MIN_SPEED_WEIGHT, slow_leg_loss),
            (config.LEG_MAX_SPEED_WEIGHT, speed_loss),
            (config.FEET_TOO_HIGH_WEIGHT, feet_high_loss),
            (config.VIRTUAL_HIP_ANGLE_LIMIT_WEIGHT, hip_angle_loss),
            (config.VIRTUAL_HIP_ELEVATION_WEIGHT, hip_elevation_loss),
            (config.LEG_GROUND_DIRECTION_WEIGHT, ground_direction_loss),
        )
        # Bound each auxiliary term so even the exponential speed penalty or
        # an awkward ground pose cannot outweigh matching the recorded gait.
        regularization = config.LEG_REGULARIZATION_MAX * sum(
            weight * (value / (1.0 + value)) for weight, value in auxiliary
        ) / max(sum(weight for weight, _ in auxiliary), 1e-8)
        total = config.LEG_ANGULAR_FIT_WEIGHT * (omega_fit + 0.25 * rate_fit) + regularization
    return total, {
        "center_velocity": center_loss,
        "angular_velocity": omega_fit,
        "leg_regularization": regularization,
        "angular_acceleration_cap": accel_loss,
        "angular_speed_cap": speed_loss,
        "legs_too_slow": slow_leg_loss,
        "belly_too_low": belly_low_loss,
        "feet_too_high": feet_high_loss,
        "front_normal_turn_cap": normal_turn_loss,
        "rear_normal_parallel": rear_normal_loss,
        "virtual_hip_angle_limits": hip_angle_loss,
        "virtual_hip_elevation_limits": hip_elevation_loss,
        "leg_ground_direction": ground_direction_loss,
        "heading_alignment": heading_alignment_loss,
        "head_ground_parallel": head_ground_parallel_loss,
        "heading_turn_progress": heading_turn_progress_loss,
    }


def _train_head(head, pairs, input_mean, input_scale, caps, geometry_prior):
    output_size = config.BODY_FEATURES if head == "body" else config.LEG_FEATURES
    output_arrays = [pair[1] if head == "body" else pair[2] for pair in pairs]
    train_values = np.concatenate([
        values[:int(len(values) * config.TRAIN_FRACTION)] for values in output_arrays
    ], axis=0)
    output_mean = train_values.mean(axis=0).astype(np.float32)
    output_scale = train_values.std(axis=0).clip(1e-3).astype(np.float32)
    train_data, val_data = _make_windows(
        pairs, input_mean, input_scale, output_mean, output_scale, head
    )
    train_x, train_y, train_pose = train_data
    val_x, val_y, val_pose = val_data
    in_mean = torch.tensor(input_mean, device=DEVICE).view(1, 1, -1)
    in_scale = torch.tensor(input_scale, device=DEVICE).view(1, 1, -1)
    out_mean = torch.tensor(output_mean, device=DEVICE).view(1, 1, -1)
    out_scale = torch.tensor(output_scale, device=DEVICE).view(1, 1, -1)

    model = MotionGRU(output_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    best_loss, best_state, wait = float("inf"), None, 0
    print(f"{head}: train windows={len(train_x)} validation={len(val_x)}")
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train_x), device=DEVICE)
        train_sum = 0.0
        for start in range(0, len(order), config.BATCH_SIZE):
            batch = order[start:start + config.BATCH_SIZE]
            prediction = model(train_x[batch])
            loss, _ = _loss(
                prediction, train_y[batch], train_x[batch, -1], in_mean, in_scale,
                out_mean, out_scale, caps, head, geometry_prior, train_pose[batch],
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_sum += float(loss.detach()) * len(batch)
        model.eval()
        with torch.no_grad():
            prediction = model(val_x)
            val_loss, components = _loss(
                prediction, val_y, val_x[:, -1], in_mean, in_scale,
                out_mean, out_scale, caps, head, geometry_prior, val_pose,
            )
        if float(val_loss) < best_loss:
            best_loss = float(val_loss)
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{head} epoch {epoch:03d} | train {train_sum / len(train_x):.4f} "
                f"| val {float(val_loss):.4f} "
                f"| omega {float(components['angular_velocity']):.4f} "
                f"| leg-aux {float(components['leg_regularization']):.4f} "
                f"| accel {float(components['angular_acceleration_cap']):.4f} "
                f"| speed-high {float(components['angular_speed_cap']):.4f} "
                f"| speed-low {float(components['legs_too_slow']):.4f} "
                f"| belly-low {float(components['belly_too_low']):.4f} "
                f"| feet-high {float(components['feet_too_high']):.4f} "
                f"| normal-turn {float(components['front_normal_turn_cap']):.4f} "
                f"| rear-normal {float(components['rear_normal_parallel']):.4f} "
                f"| virtual-hip {float(components['virtual_hip_angle_limits']):.4f} "
                f"| hip-up/down {float(components['virtual_hip_elevation_limits']):.4f} "
                f"| leg-ground {float(components['leg_ground_direction']):.4f} "
                f"| heading {float(components['heading_alignment']):.4f} "
                f"| head-level {float(components['head_ground_parallel']):.4f} "
                f"| turn-progress {float(components['heading_turn_progress']):.4f}"
            )
        if wait >= config.EARLY_STOPPING_PATIENCE:
            break

    model.load_state_dict(best_state)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.BODY_MODEL_PATH if head == "body" else config.LEGS_MODEL_PATH
    torch.save({
        "head": head,
        "model_state": model.state_dict(),
        "input_mean": input_mean,
        "input_scale": input_scale,
        "output_mean": output_mean,
        "output_scale": output_scale,
        "input_size": config.INPUT_FEATURES,
        "output_size": output_size,
        "hidden_size": config.HIDDEN_SIZE,
        "layers": config.GRU_LAYERS,
        "dropout": config.DROPOUT,
        "history_n": config.HISTORY_N,
        "prediction_m": config.PREDICTION_M,
        "generated_steps": config.GENERATED_STEPS,
        "train_prefix": config.TRAIN_PREFIX,
        "caps": caps,
        "datasets": [pair[0] for pair in pairs],
    }, checkpoint_path)
    print(f"saved {checkpoint_path}")


def main():
    pairs = _load_motion_pairs()
    all_inputs = [
        np.concatenate((body, legs), axis=1)[:int(len(body) * config.TRAIN_FRACTION)]
        for _, body, legs in pairs
    ]
    train_inputs = np.concatenate(all_inputs, axis=0)
    input_mean = train_inputs.mean(axis=0).astype(np.float32)
    input_scale = train_inputs.std(axis=0).clip(1e-3).astype(np.float32)
    caps = _measure_caps(pairs)
    morphology_path = config.TRANSLATE_DIR / "morphology.json"
    if not morphology_path.exists():
        raise FileNotFoundError("Stage 0 must run first to estimate anatomy and height limits")
    geometry_prior = json.loads(morphology_path.read_text(encoding="utf-8"))
    print(
        f"training={config.TRAIN_PREFIX} ({[pair[0] for pair in pairs]}), "
        f"N={config.HISTORY_N}, M={config.PREDICTION_M}, "
        f"frames={config.GENERATED_STEPS}, features={config.INPUT_FEATURES}; "
        f"speed cap={caps['max_angular_speed']:.4f}, "
        f"acceleration cap={caps['max_angular_acceleration']:.4f} rad/frame²"
    )
    _train_head("body", pairs, input_mean, input_scale, caps, geometry_prior)
    _train_head("legs", pairs, input_mean, input_scale, caps, geometry_prior)


if __name__ == "__main__":
    main()
