"""Train seven independent motion models on the four tracked beetle datasets."""

from pathlib import Path
import copy
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_GLOB = "*_smoothed_coords.csv"
CHECKPOINT_PATH = ROOT / "multi_beetle_gru.pt"
N_STEPS = 100
PREDICTION_STEPS = 7
TRAIN_RATIO = 0.8
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.1
EPOCHS = 60
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
PATIENCE = 12
SEED = 42

BODY_POINTS = (1, 2, 3, 4, 5)
BODY_CENTER = 1
BODY_CROSS_PAIRS = ((2, 4), (2, 5), (3, 4), (3, 5))
BODY_EDGE_PAIRS = ((1, 2), (1, 3), (2, 3), (1, 4), (1, 5), (4, 5))
LEG_ROOT_ANCHORS = (5, 3, 3, 2, 2, 4)
LEG_CHAINS = (
    (11, 17, 23),
    (9, 15, 21),
    (7, 13, 19),
    (6, 12, 18),
    (8, 14, 20),
    (10, 16, 22),
)
GROUPS = {"body": BODY_POINTS} | {
    f"leg{i + 1}": chain for i, chain in enumerate(LEG_CHAINS)
}

MIN_LEG_ANGLE_DEGREES = 30.0
MIN_BODY_ANGLE_DEGREES = 100.0
ROOT_DISTANCE_MARGIN = 0.2

# The two motion caps are the highest-priority terms in the loss.
POSITION_JUMP_CAP_WEIGHT = 100.0
ANGLE_JUMP_CAP_WEIGHT = 100.0
BEND_TURN_CAP_WEIGHT = 50.0
POSITION_MOTION_MATCH_WEIGHT = 2.0
ANGLE_MOTION_MATCH_WEIGHT = 2.0
COORDINATE_WEIGHT = 2.0
LENGTH_WEIGHT = 5000.0
ANGLE_FIT_WEIGHT = 2.0
BEND_DIRECTION_WEIGHT = 10.0
MIN_ANGLE_WEIGHT = 30.0
BODY_ANGLE_WEIGHT = 15.0
ROOT_DISTANCE_WEIGHT = 2500.0
INTER_LEG_ANGLE_WEIGHT = 0.25

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    torch.set_num_threads(2)


def expected_columns():
    return [f"pt{i}_{axis}" for i in range(1, 24) for axis in "XYZ"]


def load_subject(path):
    frame = pd.read_csv(path, sep=None, engine="python")
    frame.columns = frame.columns.astype(str).str.strip()
    columns = expected_columns()
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing[:6]}")
    coordinates = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float32
    )
    if not np.isfinite(coordinates).all():
        raise ValueError(f"{path.name}: missing or nonnumeric coordinates")
    return coordinates.reshape(-1, 23, 3), columns


def canonical_frame(coordinates):
    """Remove camera translation and align each sequence to its body point map."""
    median = np.median(coordinates, axis=0)
    longitudinal = 0.5 * (median[3] + median[4] - median[1] - median[2])
    lateral = 0.5 * (median[1] - median[2] + median[3] - median[4])
    longitudinal /= max(float(np.linalg.norm(longitudinal)), 1e-8)
    lateral -= np.dot(lateral, longitudinal) * longitudinal
    lateral /= max(float(np.linalg.norm(lateral)), 1e-8)
    normal = np.cross(longitudinal, lateral)
    normal /= max(float(np.linalg.norm(normal)), 1e-8)
    basis = np.stack((longitudinal, lateral, normal), axis=1).astype(np.float32)
    origin = coordinates[0, BODY_CENTER - 1].astype(np.float32)
    canonical = (coordinates - origin) @ basis
    return canonical.astype(np.float32), origin, basis


def angle_between(left, right):
    cross = torch.linalg.vector_norm(torch.linalg.cross(left, right, dim=-1), dim=-1)
    return torch.atan2(cross, (left * right).sum(dim=-1))


def body_angles(body):
    center = body[..., BODY_CENTER - 1, :]
    values = []
    for first, second in BODY_CROSS_PAIRS:
        a = body[..., first - 1, :] - center
        b = body[..., second - 1, :] - center
        values.append(angle_between(a, b).unsqueeze(-1))
    return torch.cat(values, dim=-1)


def body_edge_lengths(body):
    return torch.stack([
        torch.linalg.vector_norm(
            body[..., first - 1, :] - body[..., second - 1, :], dim=-1
        )
        for first, second in BODY_EDGE_PAIRS
    ], dim=-1)


def leg_angle(leg):
    return angle_between(leg[..., 0, :] - leg[..., 1, :],
                         leg[..., 2, :] - leg[..., 1, :]).unsqueeze(-1)


def leg_normal(leg):
    normal = torch.linalg.cross(
        leg[..., 0, :] - leg[..., 1, :],
        leg[..., 2, :] - leg[..., 1, :],
        dim=-1,
    )
    return normal / torch.linalg.vector_norm(
        normal, dim=-1, keepdim=True
    ).clamp_min(1e-8)


def measure_motion_caps(path):
    coordinates, _ = load_subject(path)
    canonical, _, _ = canonical_frame(coordinates)
    point_steps = np.linalg.norm(np.diff(canonical, axis=0), axis=-1)
    max_position_step = max(float(point_steps.max()), 1e-3)
    angles = []
    normal_turns = []
    for chain in LEG_CHAINS:
        leg = canonical[:, [point - 1 for point in chain], :]
        a, b, c = leg[:, 0], leg[:, 1], leg[:, 2]
        angle = np.arctan2(
            np.linalg.norm(np.cross(a - b, c - b), axis=-1),
            np.sum((a - b) * (c - b), axis=-1),
        )
        angles.append(angle)
        normals = np.cross(a - b, c - b)
        normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8)
        normal_dot = np.sum(normals[1:] * normals[:-1], axis=-1).clip(-1.0, 1.0)
        normal_turns.extend(np.arccos(normal_dot).tolist())
    body = canonical[:, :5]
    center = body[:, 0]
    for first, second in BODY_CROSS_PAIRS:
        a = body[:, first - 1] - center
        b = body[:, second - 1] - center
        angles.append(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1),
                                 np.sum(a * b, axis=-1)))
    max_angle_step = max(
        float(np.max(np.abs(np.diff(values)))) for values in angles
    )
    max_normal_turn = max(normal_turns, default=0.0)
    return {
        "source_file": path.name,
        "max_point_step": max_position_step,
        "max_angle_step_radians": max(max_angle_step, 1e-3),
        "max_bend_normal_turn_radians": max(float(max_normal_turn), 1e-3),
    }


def subject_geometry(canonical, train_end):
    source = canonical[:train_end]
    lengths = []
    bounds = []
    for chain in LEG_CHAINS:
        a, b, c = (source[:, point - 1] for point in chain)
        lengths.append((
            float(np.median(np.linalg.norm(b - a, axis=-1))),
            float(np.median(np.linalg.norm(c - b, axis=-1))),
        ))
        anchor_point = LEG_ROOT_ANCHORS[len(lengths) - 1]
        anchor = source[:, anchor_point - 1]
        distance = np.linalg.norm(a - anchor, axis=-1)
        pad = ROOT_DISTANCE_MARGIN * float(np.median(distance))
        bounds.append((
            max(0.0, float(np.percentile(distance, 5)) - pad),
            float(np.percentile(distance, 95)) + pad,
        ))
    body_lengths = [
        float(np.median(np.linalg.norm(
            source[:, first - 1] - source[:, second - 1], axis=-1
        )))
        for first, second in BODY_EDGE_PAIRS
    ]
    return {
        "leg_lengths": lengths,
        "body_edge_lengths": body_lengths,
        "root_distance_bounds": bounds,
    }


def make_windows(subjects, scaler):
    train_x, train_y, train_ids = [], [], []
    val_x, val_y, val_ids = [], [], []
    for subject_id, subject in enumerate(subjects):
        data = scaler.transform(subject["canonical"].reshape(-1, 3)).reshape(-1, 23, 3)
        data = data.astype(np.float32)
        split = subject["split"]
        flat = data.reshape(len(data), 69)
        for start in range(N_STEPS, split - PREDICTION_STEPS + 1):
            train_x.append(flat[start - N_STEPS:start])
            train_y.append(data[start:start + PREDICTION_STEPS])
            train_ids.append(subject_id)
        for start in range(max(N_STEPS, split), len(data) - PREDICTION_STEPS + 1):
            val_x.append(flat[start - N_STEPS:start])
            val_y.append(data[start:start + PREDICTION_STEPS])
            val_ids.append(subject_id)

    def pack(xs, ys, ids):
        return (
            torch.tensor(np.asarray(xs), dtype=torch.float32, device=DEVICE),
            torch.tensor(np.asarray(ys), dtype=torch.float32, device=DEVICE),
            torch.tensor(ids, dtype=torch.long, device=DEVICE),
        )

    return pack(train_x, train_y, train_ids), pack(val_x, val_y, val_ids)


class IndependentGroupGRU(nn.Module):
    """One independently trained network for the body or one leg."""

    def __init__(self, point_indices, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.point_indices = tuple(int(i) for i in point_indices)
        self.point_count = len(self.point_indices)
        self.gru = nn.GRU(
            input_size=69,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, PREDICTION_STEPS * self.point_count * 3),
        )

    def forward(self, x):
        encoded, _ = self.gru(x)
        last = x[:, -1].reshape(-1, 23, 3)
        indices = torch.tensor(self.point_indices, device=x.device)
        base = last.index_select(1, indices)
        delta = self.head(encoded[:, -1]).reshape(
            -1, PREDICTION_STEPS, self.point_count, 3
        )
        return base.unsqueeze(1) + delta


def sequence_deltas(values, previous):
    return torch.cat((values[:, :1] - previous.unsqueeze(1),
                      values[:, 1:] - values[:, :-1]), dim=1)


def project_body_group(body, edge_lengths=None):
    """Keep the two body triangles rigid, then separate them to meet angle limits."""
    if edge_lengths is None:
        return body
    eps = 1e-8
    center = body[..., 0, :]

    def make_triangle(first_i, second_i, lengths):
        # Rebuild a triangle from its three edge lengths while retaining the
        # predicted direction of its first spoke and the side of its plane.
        r1 = body[..., first_i, :] - center
        r2 = body[..., second_i, :] - center
        u1 = r1 / torch.linalg.vector_norm(r1, dim=-1, keepdim=True).clamp_min(eps)
        normal = torch.linalg.cross(r1, r2, dim=-1)
        normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp_min(eps)
        basis = nn.functional.one_hot(u1.abs().argmin(-1), 3).to(body.dtype)
        fallback = torch.linalg.cross(u1, basis, dim=-1)
        fallback = fallback / torch.linalg.vector_norm(fallback, dim=-1, keepdim=True).clamp_min(eps)
        normal = torch.where(
            (torch.linalg.vector_norm(torch.linalg.cross(r1, r2, dim=-1), dim=-1, keepdim=True) > eps),
            normal, fallback,
        )
        u2 = torch.linalg.cross(normal, u1, dim=-1)
        l1, l2, opposite = [lengths[..., i] for i in range(3)]
        while l1.ndim < u1.ndim - 1:
            l1, l2, opposite = (v.unsqueeze(-1) for v in (l1, l2, opposite))
        cos_angle = ((l1.square() + l2.square() - opposite.square()) /
                     (2.0 * l1 * l2).clamp_min(eps)).clamp(-1.0, 1.0)
        second_direction = cos_angle.unsqueeze(-1) * u1 + torch.sqrt(
            (1.0 - cos_angle.square()).clamp_min(0.0)
        ).unsqueeze(-1) * u2
        return center + u1 * l1.unsqueeze(-1), center + second_direction * l2.unsqueeze(-1)

    # Edges: (1-2, 1-3, 2-3) and (1-4, 1-5, 4-5).
    first_a, first_b = make_triangle(1, 2, edge_lengths[..., :3])
    second_a, second_b = make_triangle(3, 4, edge_lengths[..., 3:])
    ra, rb = first_a - center, first_b - center
    rc, rd = second_a - center, second_b - center
    mean_front = ra + rb
    mean_front = mean_front / torch.linalg.vector_norm(mean_front, dim=-1, keepdim=True).clamp_min(eps)
    mean_back = rc + rd
    mean_back = mean_back / torch.linalg.vector_norm(mean_back, dim=-1, keepdim=True).clamp_min(eps)
    spread_front = angle_between(ra, rb)
    spread_back = angle_between(rc, rd)
    separation = angle_between(mean_front, mean_back)
    required = body.new_tensor(np.deg2rad(MIN_BODY_ANGLE_DEGREES + 2.0)) + 0.5 * (spread_front + spread_back)
    turn = (required - separation).clamp_min(0.0)
    axis = torch.linalg.cross(mean_front, mean_back, dim=-1)
    axis_norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    basis = nn.functional.one_hot(mean_front.abs().argmin(-1), 3).to(body.dtype)
    fallback = torch.linalg.cross(mean_front, basis, dim=-1)
    fallback = fallback / torch.linalg.vector_norm(fallback, dim=-1, keepdim=True).clamp_min(eps)
    axis = torch.where(axis_norm > eps, axis / axis_norm.clamp_min(eps), fallback)
    cos_t, sin_t = torch.cos(turn).unsqueeze(-1), torch.sin(turn).unsqueeze(-1)

    def rotate(v):
        return (v * cos_t + torch.linalg.cross(axis, v, dim=-1) * sin_t
                + axis * (axis * v).sum(-1, keepdim=True) * (1.0 - cos_t))

    points = list(torch.unbind(body, dim=-2))
    points[1], points[2] = first_a, first_b
    points[3], points[4] = center + rotate(rc), center + rotate(rd)
    return torch.stack(points, dim=-2)


def project_leg_group(leg, lengths):
    """Hard projection of one proximal-joint-distal leg chain."""
    a, b_raw, c_raw = leg.unbind(dim=-2)
    eps = 1e-8
    ab = b_raw - a
    ab_len = torch.linalg.vector_norm(ab, dim=-1, keepdim=True)
    ab_fallback = torch.zeros_like(ab)
    ab_fallback[..., 0] = 1.0
    ab_dir = torch.where(ab_len > eps, ab / ab_len.clamp_min(eps), ab_fallback)
    if lengths.ndim == 1:
        first_length, second_length = lengths[0], lengths[1]
    else:
        while lengths.ndim < ab_len.ndim:
            lengths = lengths.unsqueeze(1)
        first_length, second_length = lengths[..., 0:1], lengths[..., 1:2]
    b = a + ab_dir * first_length
    bc = c_raw - b_raw
    bc_len = torch.linalg.vector_norm(bc, dim=-1, keepdim=True)
    bc_fallback = torch.zeros_like(bc)
    bc_fallback[..., 1] = 1.0
    bc_dir = torch.where(bc_len > eps, bc / bc_len.clamp_min(eps), bc_fallback)
    toward_a = -ab_dir
    angle = angle_between(toward_a, bc_dir)
    cross = torch.linalg.cross(toward_a, bc_dir, dim=-1)
    cross_len = torch.linalg.vector_norm(cross, dim=-1, keepdim=True)
    basis = nn.functional.one_hot(toward_a.abs().argmin(-1), 3).to(leg.dtype)
    fallback_axis = torch.linalg.cross(toward_a, basis, dim=-1)
    fallback_axis = fallback_axis / torch.linalg.vector_norm(
        fallback_axis, dim=-1, keepdim=True
    ).clamp_min(eps)
    axis = torch.where(cross_len > eps, cross / cross_len.clamp_min(eps), fallback_axis)
    # Small numerical safety margin so exported float coordinates stay >= 30°.
    min_angle = leg.new_tensor(np.deg2rad(MIN_LEG_ANGLE_DEGREES + 0.5))
    delta = (min_angle - angle).clamp_min(0.0)
    cd, sd = torch.cos(delta).unsqueeze(-1), torch.sin(delta).unsqueeze(-1)
    bc_fixed = (bc_dir * cd + torch.linalg.cross(axis, bc_dir, dim=-1) * sd
                + axis * (axis * bc_dir).sum(-1, keepdim=True) * (1.0 - cd))
    c = b + bc_fixed * second_length
    return torch.stack((a, b, c), dim=-2)


def project_full_pose(pose, lengths, bounds, body_edge_lengths):
    """Apply leg lengths/angles, broad root-distance limits and body angles."""
    projected = pose.clone()
    fixed_body = project_body_group(projected[..., :5, :], body_edge_lengths)
    updated = list(torch.unbind(projected, dim=-2))
    for i in range(5):
        updated[i] = fixed_body[..., i, :]
    projected = torch.stack(updated, dim=-2)
    for leg_index, chain in enumerate(LEG_CHAINS):
        indices = torch.tensor([p - 1 for p in chain], device=pose.device)
        leg = projected.index_select(-2, indices)
        fixed = project_leg_group(leg, lengths[leg_index])
        updated = list(torch.unbind(projected, dim=-2))
        for local_i, point_i in enumerate(indices.tolist()):
            updated[point_i] = fixed[..., local_i, :]
        projected = torch.stack(updated, dim=-2)

        root = projected[..., chain[0] - 1, :]
        anchor = projected[..., LEG_ROOT_ANCHORS[leg_index] - 1, :]
        offset = root - anchor
        distance = torch.linalg.vector_norm(offset, dim=-1, keepdim=True)
        direction = offset / distance.clamp_min(1e-8)
        low, high = bounds[leg_index]
        # Stay just inside the learned interval to absorb CSV float rounding.
        margin = 1e-3
        clamped = distance.clamp(min=float(low) + margin, max=float(high) - margin)
        shift = direction * (clamped - distance)
        updated = list(torch.unbind(projected, dim=-2))
        for point in chain:
            updated[point - 1] = updated[point - 1] + shift
        projected = torch.stack(updated, dim=-2)

    return projected


def limit_leg_frame_change(previous_pose, candidate_pose, lengths, max_angle_step,
                          max_normal_turn):
    """Keep each leg's joint angle and bend plane within the sampled frame limits."""
    result = candidate_pose.clone()
    angle_cap = float(max_angle_step)
    normal_cap = float(max_normal_turn)
    for leg_index, chain in enumerate(LEG_CHAINS):
        indices = torch.tensor([p - 1 for p in chain], device=result.device)
        previous_leg = previous_pose.index_select(-2, indices)
        current_leg = result.index_select(-2, indices)
        current_leg = project_leg_group(current_leg, lengths[leg_index])

        a, b, c = current_leg.unbind(dim=-2)
        pa, pb, pc = previous_leg.unbind(dim=-2)
        root_dir = (b - a) / torch.linalg.vector_norm(b - a).clamp_min(1e-8)
        toward_root = -root_dir
        previous_angle = angle_between(pa - pb, pc - pb)
        current_angle = angle_between(toward_root, c - b)
        min_angle = np.deg2rad(MIN_LEG_ANGLE_DEGREES + 0.5)
        lower = max(min_angle, float(previous_angle) - angle_cap)
        upper = min(np.pi, float(previous_angle) + angle_cap)
        desired_angle = current_angle.clamp(min=lower, max=upper)

        v = (c - b) / torch.linalg.vector_norm(c - b).clamp_min(1e-8)
        current_normal = torch.linalg.cross(toward_root, v, dim=-1)
        current_normal = current_normal / torch.linalg.vector_norm(current_normal).clamp_min(1e-8)
        previous_normal = torch.linalg.cross(pa - pb, pc - pb, dim=-1)
        previous_normal = previous_normal / torch.linalg.vector_norm(previous_normal).clamp_min(1e-8)
        previous_normal = previous_normal - torch.dot(previous_normal, toward_root) * toward_root
        previous_normal = previous_normal / torch.linalg.vector_norm(previous_normal).clamp_min(1e-8)
        normal_angle = torch.acos(torch.dot(previous_normal, current_normal).clamp(-1.0, 1.0))
        if float(normal_angle) > normal_cap:
            amount = normal_cap / float(normal_angle)
            sine = torch.sin(normal_angle).clamp_min(1e-8)
            chosen_normal = (
                torch.sin((1.0 - amount) * normal_angle) / sine * previous_normal
                + torch.sin(amount * normal_angle) / sine * current_normal
            )
        else:
            chosen_normal = current_normal
        chosen_normal = chosen_normal / torch.linalg.vector_norm(chosen_normal).clamp_min(1e-8)
        bend_axis = torch.linalg.cross(chosen_normal, toward_root, dim=-1)
        bend_axis = bend_axis / torch.linalg.vector_norm(bend_axis).clamp_min(1e-8)
        new_direction = torch.cos(desired_angle) * toward_root + torch.sin(desired_angle) * bend_axis
        new_c = b + new_direction * lengths[leg_index, 1]

        updated = list(torch.unbind(result, dim=-2))
        updated[chain[0] - 1] = a
        updated[chain[1] - 1] = b
        updated[chain[2] - 1] = new_c
        result = torch.stack(updated, dim=-2)
    return result


def pose_angles(pose):
    values = [body_angles(pose[..., :5, :])]
    for chain in LEG_CHAINS:
        indices = torch.tensor([p - 1 for p in chain], device=pose.device)
        values.append(leg_angle(pose.index_select(-2, indices)))
    return torch.cat(values, dim=-1)


def cap_frame_motion(previous, candidate, caps, search_steps=20):
    """Prioritize per-point and per-angle frame-to-frame caps at export time."""
    previous_angles = pose_angles(previous)
    previous_normals = []
    for chain in LEG_CHAINS:
        indices = torch.tensor([p - 1 for p in chain], device=previous.device)
        previous_normals.append(leg_normal(previous.index_select(-2, indices)))
    previous_normals = torch.stack(previous_normals, dim=-2)
    delta = candidate - previous

    def is_safe(amount):
        pose = previous + amount * delta
        max_point_step = torch.linalg.vector_norm(pose - previous, dim=-1).max()
        max_angle_step = (pose_angles(pose) - previous_angles).abs().max()
        current_normals = []
        for chain in LEG_CHAINS:
            indices = torch.tensor([p - 1 for p in chain], device=pose.device)
            current_normals.append(leg_normal(pose.index_select(-2, indices)))
        current_normals = torch.stack(current_normals, dim=-2)
        normal_turn = torch.acos(
            (current_normals * previous_normals).sum(-1).clamp(-1.0, 1.0)
        ).max()
        return (
            float(max_point_step) <= caps["max_point_step"] + 1e-5
            and float(max_angle_step) <= caps["max_angle_step"] + 1e-5
            and float(normal_turn) <= caps["max_bend_normal_turn"] + 1e-5
        )

    if is_safe(1.0):
        return candidate
    low, high = 0.0, 1.0
    for _ in range(search_steps):
        middle = 0.5 * (low + high)
        if is_safe(middle):
            low = middle
        else:
            high = middle
    return previous + low * delta


def inter_leg_angles(full_pose):
    center = full_pose[..., BODY_CENTER - 1, :]
    legs = torch.stack([
        torch.stack([full_pose[..., p - 1, :] for p in chain], dim=-2)
        for chain in LEG_CHAINS
    ], dim=-3)
    rays = legs - center.unsqueeze(-2).unsqueeze(-2)
    rays = rays / torch.linalg.vector_norm(
        rays, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    return torch.stack([
        angle_between(rays[..., i, :, :], rays[..., j, :, :])
        for i in range(6) for j in range(i + 1, 6)
    ], dim=-1)


def loss_for_group(name, pred_scaled, target_full_scaled, previous_full_scaled,
                   subject_ids, scaler, lengths_by_subject,
                   body_lengths_by_subject, bounds_by_subject, caps):
    point_ids = GROUPS[name]
    indices = torch.tensor([point - 1 for point in point_ids], device=DEVICE)
    mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=DEVICE).view(1, 1, 1, 3)
    scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=DEVICE).view(1, 1, 1, 3)
    target_raw = target_full_scaled * scale + mean
    previous_full = previous_full_scaled.reshape(-1, 23, 3) * scale[:, 0] + mean[:, 0]
    previous_group = previous_full.index_select(1, indices)
    target_group = target_raw.index_select(2, indices)
    raw_prediction = pred_scaled * scale + mean
    batch = len(subject_ids)
    predicted_lengths = None
    if name == "body":
        body_lengths = torch.tensor(
            body_lengths_by_subject, dtype=torch.float32, device=DEVICE
        )[subject_ids]
        raw_body_lengths = body_edge_lengths(raw_prediction)
        body_length_error = (
            raw_body_lengths - body_lengths[:, None, :]
        ) / body_lengths[:, None, :].clamp_min(1e-6)
        length_loss = body_length_error.square().mean()
        projected_group = project_body_group(raw_prediction, body_lengths)
        predicted_angles = body_angles(projected_group)
        target_angles = body_angles(target_group)
        previous_angles = body_angles(previous_group)
        raw_angles = body_angles(raw_prediction)
        angle_floor = raw_prediction.new_tensor(np.deg2rad(MIN_BODY_ANGLE_DEGREES))
        body_limit = torch.relu(angle_floor - raw_angles).square().mean()
        bend_loss = raw_prediction.new_zeros(())
        min_angle = raw_prediction.new_zeros(())
    else:
        leg_index = int(name[3:]) - 1
        lengths = torch.tensor(lengths_by_subject, dtype=torch.float32, device=DEVICE)[
            subject_ids, leg_index
        ]
        predicted_lengths = torch.stack((
            torch.linalg.vector_norm(raw_prediction[..., 1, :] - raw_prediction[..., 0, :], dim=-1),
            torch.linalg.vector_norm(raw_prediction[..., 2, :] - raw_prediction[..., 1, :], dim=-1),
        ), dim=-1)
        length_error = (predicted_lengths - lengths[:, None, :]) / lengths[:, None, :].clamp_min(1e-6)
        length_loss = length_error.square().mean()
        raw_angles = leg_angle(raw_prediction)
        angle_floor = raw_prediction.new_tensor(np.deg2rad(MIN_LEG_ANGLE_DEGREES))
        min_angle = torch.relu(angle_floor - raw_angles).square().mean()
        projected_group = project_leg_group(raw_prediction, lengths)
        predicted_angles = leg_angle(projected_group)
        target_angles = leg_angle(target_group)
        previous_angles = leg_angle(previous_group)
        pred_n = leg_normal(projected_group)
        target_n = leg_normal(target_group)
        bend_loss = torch.relu(-(pred_n * target_n).sum(dim=-1)).square().mean()
        body_limit = raw_prediction.new_zeros(())

    predicted_delta = sequence_deltas(raw_prediction, previous_group)
    target_delta = sequence_deltas(target_group, previous_group)
    position_step = torch.linalg.vector_norm(predicted_delta, dim=-1)
    position_excess = torch.relu(position_step - caps["max_point_step"])
    position_cap_loss = (position_excess / caps["max_point_step"]).square().mean()
    position_match_loss = (
        (predicted_delta - target_delta) / caps["max_point_step"]
    ).square().mean()

    predicted_angle_delta = sequence_deltas(predicted_angles, previous_angles)
    target_angle_delta = sequence_deltas(target_angles, previous_angles)
    angle_excess = torch.relu(
        predicted_angle_delta.abs() - caps["max_angle_step"]
    )
    angle_cap_loss = (angle_excess / caps["max_angle_step"]).square().mean()
    angle_match_loss = (
        (predicted_angle_delta - target_angle_delta) / caps["max_angle_step"]
    ).square().mean()

    bend_turn_cap_loss = raw_prediction.new_zeros(())
    if name != "body":
        normals = leg_normal(projected_group)
        previous_normal = leg_normal(previous_group)
        normal_delta = torch.cat((
            torch.acos((normals[:, :1] * previous_normal.unsqueeze(1)).sum(-1).clamp(-1, 1)),
            torch.acos((normals[:, 1:] * normals[:, :-1]).sum(-1).clamp(-1, 1)),
        ), dim=1)
        normal_excess = torch.relu(normal_delta - caps["max_bend_normal_turn"])
        bend_turn_cap_loss = (
            normal_excess / caps["max_bend_normal_turn"]
        ).square().mean()

    projected_scaled = (projected_group - mean) / scale
    target_group_scaled = target_full_scaled.index_select(2, indices)
    coordinate_loss = nn.functional.mse_loss(
        projected_scaled, target_group_scaled
    )
    predicted_full = target_raw.clone()
    predicted_full[:, :, indices, :] = projected_group
    interleg_loss = nn.functional.mse_loss(
        inter_leg_angles(predicted_full), inter_leg_angles(target_raw)
    )

    subject_bounds = torch.tensor(bounds_by_subject, dtype=torch.float32, device=DEVICE)[subject_ids]
    root_distances = torch.stack([
        torch.linalg.vector_norm(
            predicted_full[..., chain[0] - 1, :]
            - predicted_full[..., LEG_ROOT_ANCHORS[i] - 1, :], dim=-1
        ) for i, chain in enumerate(LEG_CHAINS)
    ], dim=-1)
    root_error = (
        torch.relu(subject_bounds[:, None, :, 0] - root_distances)
        + torch.relu(root_distances - subject_bounds[:, None, :, 1])
    )
    root_scale = torch.tensor(lengths_by_subject, dtype=torch.float32, device=DEVICE)[
        subject_ids
    ].mean(dim=(-1, -2)).clamp_min(1e-6)
    root_loss = (root_error / root_scale[:, None, None]).square().mean()

    total = (
        POSITION_JUMP_CAP_WEIGHT * position_cap_loss
        + ANGLE_JUMP_CAP_WEIGHT * angle_cap_loss
        + BEND_TURN_CAP_WEIGHT * bend_turn_cap_loss
        + POSITION_MOTION_MATCH_WEIGHT * position_match_loss
        + ANGLE_MOTION_MATCH_WEIGHT * angle_match_loss
        + COORDINATE_WEIGHT * coordinate_loss
        + LENGTH_WEIGHT * length_loss
        + ANGLE_FIT_WEIGHT * nn.functional.mse_loss(predicted_angles, target_angles)
        + BEND_DIRECTION_WEIGHT * bend_loss
        + MIN_ANGLE_WEIGHT * min_angle
        + BODY_ANGLE_WEIGHT * body_limit
        + ROOT_DISTANCE_WEIGHT * root_loss
        + INTER_LEG_ANGLE_WEIGHT * interleg_loss
    )
    components = {
        "position_cap": position_cap_loss,
        "angle_cap": angle_cap_loss,
        "bend_turn_cap": bend_turn_cap_loss,
        "position_match": position_match_loss,
        "angle_match": angle_match_loss,
        "coordinate": coordinate_loss,
        "length": length_loss,
        "angle_fit": nn.functional.mse_loss(predicted_angles, target_angles),
        "bend_direction": bend_loss,
        "min_angle": min_angle,
        "body_angle": body_limit,
        "root_distance": root_loss,
        "inter_leg": interleg_loss,
    }
    return total, predicted_angles, projected_group, components


def train_one_group(name, point_ids, train_data, val_data, scaler,
                    lengths_by_subject, body_lengths_by_subject,
                    bounds_by_subject, caps):
    model = IndependentGroupGRU([point - 1 for point in point_ids]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    tx, ty, tids = train_data
    vx, vy, vids = val_data
    best_loss = float("inf")
    best_state = None
    wait = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(tx), device=DEVICE)
        train_sum = 0.0
        first_components = None
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            pred = model(tx[batch])
            loss, _, _, components = loss_for_group(
                name, pred, ty[batch], tx[batch, -1], tids[batch], scaler,
                lengths_by_subject, body_lengths_by_subject,
                bounds_by_subject, caps,
            )
            if first_components is None:
                first_components = {
                    key: round(float(value.detach().cpu()), 4)
                    for key, value in components.items()
                }
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_sum += loss.item() * len(batch)
        model.eval()
        with torch.no_grad():
            pred = model(vx)
            val_loss, pred_angles, _, _ = loss_for_group(
                name, pred, vy, vx[:, -1], vids, scaler,
                lengths_by_subject, body_lengths_by_subject,
                bounds_by_subject, caps,
            )
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{name} epoch {epoch:03d} | train {train_sum / len(tx):.4f} "
                f"| val {val_loss.item():.4f}"
            )
        if epoch == 1:
            print(f"  first-batch loss terms: {first_components}")
        if wait >= PATIENCE:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(vx)
        _, predicted_angles, _, _ = loss_for_group(
            name, prediction, vy, vx[:, -1], vids, scaler,
            lengths_by_subject, body_lengths_by_subject,
            bounds_by_subject, caps,
        )
        group_indices = torch.tensor([p - 1 for p in point_ids], device=DEVICE)
        mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=DEVICE).view(1, 1, 1, 3)
        scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=DEVICE).view(1, 1, 1, 3)
        true_group = (vy.index_select(2, group_indices) * scale + mean)
        prev_full_scaled = vx[:, -1].reshape(-1, 23, 3)
        prev_group = (
            prev_full_scaled.index_select(1, group_indices)
            * scale[:, 0] + mean[:, 0]
        )
        if name == "body":
            target_angles = body_angles(true_group)
            baseline_angles = body_angles(prev_group).unsqueeze(1).expand_as(target_angles)
        else:
            target_angles = leg_angle(true_group)
            baseline_angles = leg_angle(prev_group).unsqueeze(1).expand_as(target_angles)
        pred_rmse = torch.mean((predicted_angles - target_angles) ** 2).sqrt().item()
        base_rmse = torch.mean((baseline_angles - target_angles) ** 2).sqrt().item()
    print(
        f"{name} validation angle RMSE: model {np.degrees(pred_rmse):.2f}°, "
        f"persistence {np.degrees(base_rmse):.2f}°"
    )
    return model.state_dict()


def main():
    paths = sorted(DATA_DIR.glob(DATA_GLOB))
    if len(paths) != 4:
        raise ValueError(f"Expected four raw beetle CSVs, found {len(paths)}: {paths}")
    columns = expected_columns()
    subjects = []
    for path in paths:
        coords, _ = load_subject(path)
        if len(coords) < N_STEPS + PREDICTION_STEPS + 3:
            raise ValueError(f"Not enough frames in {path.name}")
        canonical, origin, basis = canonical_frame(coords)
        split = int(len(coords) * TRAIN_RATIO)
        subjects.append({
            "name": path.stem.replace("_smoothed_coords", ""),
            "path": str(path),
            "coordinates": coords,
            "canonical": canonical,
            "origin": origin,
            "basis": basis,
            "split": split,
        })
        print(f"{path.name}: {len(coords)} frames; train {split}; val {len(coords)-split}")

    random_source = random.Random(SEED).choice(paths)
    caps = measure_motion_caps(random_source)
    caps = {
        "max_point_step": caps["max_point_step"],
        "max_angle_step": caps["max_angle_step_radians"],
        "max_bend_normal_turn": caps["max_bend_normal_turn_radians"],
        "source_file": caps["source_file"],
    }
    print(
        f"Случайный файл для пределов: {caps['source_file']} | "
        f"max point step {caps['max_point_step']:.3f} | "
        f"max angle step {np.degrees(caps['max_angle_step']):.2f}° | "
        f"max bend-normal turn {np.degrees(caps['max_bend_normal_turn']):.2f}°"
    )

    scaler = StandardScaler()
    fit_values = np.concatenate([
        subject["canonical"][:subject["split"]].reshape(-1, 3)
        for subject in subjects
    ])
    scaler.fit(fit_values)
    train_data, val_data = make_windows(subjects, scaler)
    print(f"Train windows: {len(train_data[0])}; validation windows: {len(val_data[0])}")

    geometries = [
        subject_geometry(subject["canonical"], subject["split"])
        for subject in subjects
    ]
    lengths_by_subject = np.asarray([g["leg_lengths"] for g in geometries], dtype=np.float32)
    body_lengths_by_subject = np.asarray(
        [g["body_edge_lengths"] for g in geometries], dtype=np.float32
    )
    bounds_by_subject = np.asarray([g["root_distance_bounds"] for g in geometries], dtype=np.float32)
    print("Целевые длины рёбер тела:", np.round(body_lengths_by_subject, 2).tolist())
    print("Диапазоны от корней ног до точек тела:", np.round(bounds_by_subject, 2).tolist())

    states = {}
    for name, points in GROUPS.items():
        print(f"\nОбучаю независимую модель: {name}, точки {points}")
        states[name] = train_one_group(
            name, points, train_data, val_data, scaler,
            lengths_by_subject, body_lengths_by_subject,
            bounds_by_subject, caps,
        )

    torch.save({
        "model_states": states,
        "groups": GROUPS,
        "n_steps": N_STEPS,
        "prediction_steps": PREDICTION_STEPS,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "subject_names": [subject["name"] for subject in subjects],
        "subject_geometries": {
            subject["name"]: geometry
            for subject, geometry in zip(subjects, geometries)
        },
        "motion_caps": caps,
        "canonicalization": "origin=first_pt1; axes from median body groups pt2/3 and pt4/5",
    }, CHECKPOINT_PATH)
    print(f"\nSaved seven independent models: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
