
from pathlib import Path
import copy
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler


# ==================== НАСТРОЙКИ ====================

CSV_PATH = "data/Scarites_6_smoothed_coords.csv"

# Количество предыдущих кадров для предсказания следующего.
N_STEPS = 100
PREDICTION_STEPS = 7

HIDDEN_SIZE = 128
NUM_LAYERS = 3
DROPOUT = 0.1

EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
PATIENCE = 30
TRAIN_RATIO = 0.8

SEED = 42
MODEL_PATH = "scarites_gru.pt"
OUTPUT_PATH = "data/Scarites_6_prediction.csv"

# Точки перечислены в ANATOMY.txt (номера в файле начинаются с 1).
BODY_POINTS = (1, 2, 3, 4, 5)
# Все цепочки ног теперь идут одинаково: от корня у тела к концу ноги.
LEG_POINTS = (
    (11, 17, 23),
    (9, 15, 21),
    (7, 13, 19),
    (6, 12, 18),
    (8, 14, 20),
    (10, 16, 22),
)
LEG_LENGTH_LOSS_WEIGHT = 100.0
LEG_ANGLE_LOSS_WEIGHT = 2.0
LEG_BEND_DIRECTION_LOSS_WEIGHT = 10.0
MIN_LEG_ANGLE_LOSS_WEIGHT = 30.0
BODY_ANGLE_LIMIT_LOSS_WEIGHT = 15.0
INTER_LEG_ANGLE_LOSS_WEIGHT = 0.25
LEG_BODY_DISTANCE_LOSS_WEIGHT = 5.0
ANGLE_CHANGE_LOSS_WEIGHT = 1.0
ANTI_FREEZE_LOSS_WEIGHT = 0.25
ROOT_MOTION_LOSS_WEIGHT = 0.1
COORDINATE_LOSS_WEIGHT = 2.0
COORDINATE_CHANGE_LOSS_WEIGHT = 1.0
MIN_LEG_ANGLE_DEGREES = 30.0
MIN_BODY_CROSS_ANGLE_DEGREES = 100.0
BODY_CENTER_POINT = 1
# Angles across the two body triangles (pt2/pt3 against pt4/pt5).
BODY_CROSS_PAIRS = ((2, 4), (2, 5), (3, 4), (3, 5))
LEG_ROOT_DISTANCE_MARGIN = 0.75

# ==================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Устройство:", DEVICE)


def load_data(path):
    # sep=None позволяет определить запятые или табуляцию.
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = df.columns.astype(str).str.strip()

    expected = [
        f"pt{i}_{axis}"
        for i in range(1, 24)
        for axis in ("X", "Y", "Z")
    ]

    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"Не найдены столбцы: {missing}\n"
            f"Первые столбцы файла: {list(df.columns[:10])}"
        )

    coords = df[expected].apply(
        pd.to_numeric, errors="coerce"
    )

    if coords.isna().any().any():
        raise ValueError(
            "В координатах есть пропуски или нечисловые значения. "
            "Проверь исходный CSV."
        )

    # (число кадров, 23 точки, 3 координаты)
    return coords.to_numpy(dtype=np.float32).reshape(-1, 23, 3), expected


def make_windows(data, n_steps, prediction_steps):
    """Вход: N кадров; цель: следующие M кадров."""
    X, y = [], []

    for i in range(n_steps, len(data) - prediction_steps + 1):
        X.append(data[i - n_steps:i])
        y.append(data[i:i + prediction_steps])

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
    )


def angle_between(left, right):
    cross_norm = torch.linalg.vector_norm(
        torch.linalg.cross(left, right, dim=-1), dim=-1
    )
    dot = (left * right).sum(dim=-1)
    return torch.atan2(cross_norm, dot)


def leg_angles(points):
    """Внутренние углы шести ног, в порядке корень—сустав—конец."""
    result = []
    for chain in LEG_POINTS:
        a, b, c = (points[..., point - 1, :] for point in chain)
        result.append(angle_between(a - b, c - b).unsqueeze(-1))
    return torch.cat(result, dim=-1)


def body_cross_angles(points):
    """Углы у центральной pt1 между двумя половинами тела."""
    center = points[..., BODY_CENTER_POINT - 1, :]
    angles = []
    for first, second in BODY_CROSS_PAIRS:
        a = points[..., first - 1, :] - center
        b = points[..., second - 1, :] - center
        angles.append(angle_between(a, b).unsqueeze(-1))
    return torch.cat(angles, dim=-1)


def joint_angles(points):
    """Все контролируемые углы: четыре угла тела и шесть углов ног."""
    return torch.cat((body_cross_angles(points), leg_angles(points)), dim=-1)


def leg_segment_lengths(points):
    lengths = []
    for chain in LEG_POINTS:
        a, b, c = (points[..., point - 1, :] for point in chain)
        lengths.append(torch.stack((
            torch.linalg.vector_norm(b - a, dim=-1),
            torch.linalg.vector_norm(c - b, dim=-1),
        ), dim=-1))
    return torch.stack(lengths, dim=-2)


def leg_bend_orientation_loss(predicted, target):
    """Не допускает переворота плоскости сгиба относительно размеченного кадра."""
    penalties = []
    eps = 1e-8
    for chain in LEG_POINTS:
        a, b, c = (predicted[..., point - 1, :] for point in chain)
        ta, tb, tc = (target[..., point - 1, :] for point in chain)
        normal = torch.linalg.cross(a - b, c - b, dim=-1)
        target_normal = torch.linalg.cross(ta - tb, tc - tb, dim=-1)
        normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp_min(eps)
        target_normal = target_normal / torch.linalg.vector_norm(
            target_normal, dim=-1, keepdim=True
        ).clamp_min(eps)
        # Ноль при правильной стороне, штраф только за противоположную сторону.
        penalties.append(torch.relu(-(normal * target_normal).sum(dim=-1)).square())
    return torch.stack(penalties, dim=-1).mean()


def inter_leg_angles(points):
    """Парные углы между ногами на уровнях корня, сустава и конца."""
    center = points[..., BODY_CENTER_POINT - 1, :]
    leg_points = torch.stack([
        torch.stack([points[..., point - 1, :] for point in chain], dim=-2)
        for chain in LEG_POINTS
    ], dim=-3)  # (..., 6 legs, 3 levels, 3 coords)
    rays = leg_points - center.unsqueeze(-2).unsqueeze(-2)
    rays = rays / torch.linalg.vector_norm(rays, dim=-1, keepdim=True).clamp_min(1e-8)
    pairs = [(i, j) for i in range(len(LEG_POINTS)) for j in range(i + 1, len(LEG_POINTS))]
    return torch.stack([
        angle_between(rays[..., i, :, :], rays[..., j, :, :])
        for i, j in pairs
    ], dim=-1)


def project_body_cross_angles(points):
    """Раскрывает две пары точек тела до минимального угла вокруг pt1."""
    projected = points
    center_index = BODY_CENTER_POINT - 1
    min_angle = points.new_tensor(np.deg2rad(MIN_BODY_CROSS_ANGLE_DEGREES))
    eps = 1e-8

    # Повторные проходы учитывают, что поворот одной точки влияет на две пары.
    for _ in range(8):
        for first, second in BODY_CROSS_PAIRS:
            center = projected[..., center_index, :]
            first_index = first - 1
            second_index = second - 1
            first_ray = projected[..., first_index, :] - center
            second_ray = projected[..., second_index, :] - center
            first_norm = torch.linalg.vector_norm(first_ray, dim=-1, keepdim=True)
            second_norm = torch.linalg.vector_norm(second_ray, dim=-1, keepdim=True)
            a = first_ray / first_norm.clamp_min(eps)
            b = second_ray / second_norm.clamp_min(eps)

            cosine = (a * b).sum(dim=-1).clamp(-1.0, 1.0)
            angle = torch.acos(cosine)
            delta = (min_angle - angle).clamp_min(0.0)

            axis = torch.linalg.cross(a, b, dim=-1)
            axis_norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
            basis_index = a.abs().argmin(dim=-1)
            basis = nn.functional.one_hot(basis_index, num_classes=3).to(points.dtype)
            fallback_axis = torch.linalg.cross(a, basis, dim=-1)
            fallback_axis = fallback_axis / torch.linalg.vector_norm(
                fallback_axis, dim=-1, keepdim=True
            ).clamp_min(eps)
            axis = torch.where(
                axis_norm > eps, axis / axis_norm.clamp_min(eps), fallback_axis
            )

            cos_delta = torch.cos(delta).unsqueeze(-1)
            sin_delta = torch.sin(delta).unsqueeze(-1)
            rotated = (
                b * cos_delta
                + torch.linalg.cross(axis, b, dim=-1) * sin_delta
                + axis * (axis * b).sum(dim=-1, keepdim=True) * (1.0 - cos_delta)
            )
            updated = list(torch.unbind(projected, dim=-2))
            updated[second_index] = center + rotated * second_norm
            projected = torch.stack(updated, dim=-2)
    return projected


def project_leg_kinematics(points, reference_lengths, root_distance_bounds=None):
    """Жёстко держит длины и минимум угла; ограничивает вылет корней от тела."""
    projected = points
    min_angle = points.new_tensor(np.deg2rad(MIN_LEG_ANGLE_DEGREES))
    eps = 1e-8

    for leg_index, chain in enumerate(LEG_POINTS):
        a_idx, b_idx, c_idx = (point - 1 for point in chain)
        a = projected[..., a_idx, :]
        b_raw = projected[..., b_idx, :]
        c_raw = projected[..., c_idx, :]

        ab = b_raw - a
        ab_norm = torch.linalg.vector_norm(ab, dim=-1, keepdim=True)
        ab_fallback = torch.zeros_like(ab)
        ab_fallback[..., 0] = 1.0
        ab_dir = torch.where(ab_norm > eps, ab / ab_norm.clamp_min(eps), ab_fallback)
        b = a + ab_dir * reference_lengths[leg_index, 0]

        bc = c_raw - b_raw
        bc_norm = torch.linalg.vector_norm(bc, dim=-1, keepdim=True)
        bc_fallback = torch.zeros_like(bc)
        bc_fallback[..., 1] = 1.0
        bc_dir = torch.where(bc_norm > eps, bc / bc_norm.clamp_min(eps), bc_fallback)

        toward_a = -ab_dir
        cosine = (toward_a * bc_dir).sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        angle = torch.acos(cosine)
        cross = torch.linalg.cross(toward_a, bc_dir, dim=-1)
        cross_norm = torch.linalg.vector_norm(cross, dim=-1, keepdim=True)
        axis_regular = cross / cross_norm.clamp_min(eps)

        # При почти параллельных звеньях выбираем устойчивую перпендикулярную ось.
        basis_index = toward_a.abs().argmin(dim=-1)
        basis = nn.functional.one_hot(basis_index, num_classes=3).to(points.dtype)
        axis_fallback = torch.linalg.cross(toward_a, basis, dim=-1)
        axis_fallback = axis_fallback / torch.linalg.vector_norm(
            axis_fallback, dim=-1, keepdim=True
        ).clamp_min(eps)
        axis = torch.where(cross_norm > eps, axis_regular, axis_fallback)

        delta = (min_angle - angle).clamp_min(0.0)
        cos_delta = torch.cos(delta).unsqueeze(-1)
        sin_delta = torch.sin(delta).unsqueeze(-1)
        bc_rotated = (
            bc_dir * cos_delta
            + torch.linalg.cross(axis, bc_dir, dim=-1) * sin_delta
            + axis * (axis * bc_dir).sum(dim=-1, keepdim=True) * (1.0 - cos_delta)
        )
        c = b + bc_rotated * reference_lengths[leg_index, 1]

        # Точки цепочки могут быть перечислены в обратном порядке.
        updated = list(torch.unbind(projected, dim=-2))
        updated[b_idx] = b
        updated[c_idx] = c
        projected = torch.stack(updated, dim=-2)

    if root_distance_bounds is not None:
        center = projected[..., BODY_CENTER_POINT - 1, :]
        for leg_index, chain in enumerate(LEG_POINTS):
            indices = [point - 1 for point in chain]
            root = projected[..., indices[0], :]
            offset = root - center
            distance = torch.linalg.vector_norm(offset, dim=-1, keepdim=True)
            fallback = torch.zeros_like(offset)
            fallback[..., 0] = 1.0
            direction = torch.where(
                distance > eps, offset / distance.clamp_min(eps), fallback
            )
            lower, upper = root_distance_bounds[leg_index]
            desired = distance.clamp(min=lower, max=upper)
            shift = direction * (desired - distance)
            updated = list(torch.unbind(projected, dim=-2))
            for point_index in indices:
                updated[point_index] = updated[point_index] + shift.squeeze(-1)
            projected = torch.stack(updated, dim=-2)

    return project_body_cross_angles(projected)


def sequence_deltas(values, previous):
    return torch.cat(
        (values[:, :1] - previous.unsqueeze(1), values[:, 1:] - values[:, :-1]),
        dim=1,
    )


def motion_loss(
    pred_scaled, target_scaled, previous_scaled, mean_t, scale_t,
    leg_lengths_t, root_distance_bounds_t,
):
    # Углы и ограничения считаются в исходном пространстве координат.
    pred_raw = pred_scaled * scale_t + mean_t
    target = target_scaled * scale_t + mean_t
    previous = previous_scaled * scale_t + mean_t

    raw_leg_angles = leg_angles(pred_raw)
    angle_floor = pred_raw.new_tensor(np.deg2rad(MIN_LEG_ANGLE_DEGREES))
    min_angle_loss = torch.relu(angle_floor - raw_leg_angles).square().mean()

    raw_lengths = leg_segment_lengths(pred_raw)
    relative_length_error = (raw_lengths - leg_lengths_t) / leg_lengths_t.clamp_min(1e-6)
    length_loss = relative_length_error.square().mean()

    # Проекция гарантирует длины, минимум угла и широкий коридор корней ног.
    pred = project_leg_kinematics(
        pred_raw, leg_lengths_t, root_distance_bounds_t
    )
    pred_angles = joint_angles(pred)
    target_angles = joint_angles(target)
    angle_loss = nn.functional.mse_loss(pred_angles, target_angles)

    body_floor = pred_raw.new_tensor(np.deg2rad(MIN_BODY_CROSS_ANGLE_DEGREES))
    body_limit_loss = torch.relu(
        body_floor - body_cross_angles(pred_raw)
    ).square().mean()
    bend_direction_loss = leg_bend_orientation_loss(pred, target)
    inter_leg_angle_loss = nn.functional.mse_loss(
        inter_leg_angles(pred), inter_leg_angles(target)
    )

    center = pred_raw[..., BODY_CENTER_POINT - 1, :]
    root_distances = torch.stack([
        torch.linalg.vector_norm(pred_raw[..., chain[0] - 1, :] - center, dim=-1)
        for chain in LEG_POINTS
    ], dim=-1)
    lower = root_distance_bounds_t[:, 0]
    upper = root_distance_bounds_t[:, 1]
    root_scale = leg_lengths_t.mean(dim=-1).clamp_min(1e-6)
    root_error = (
        torch.relu(lower - root_distances) + torch.relu(root_distances - upper)
    ) / root_scale
    root_distance_loss = root_error.square().mean()

    pred_leg_angles = leg_angles(pred)
    target_leg_angles = leg_angles(target)
    previous_leg_angles = leg_angles(previous)
    pred_angle_delta = sequence_deltas(pred_leg_angles, previous_leg_angles)
    target_angle_delta = sequence_deltas(target_leg_angles, previous_leg_angles)
    angle_change_loss = nn.functional.mse_loss(pred_angle_delta, target_angle_delta)

    # Дополнительный анти-статичный штраф включается, когда размеченная нога
    # действительно сгибалась; паузы в исходных данных он не превращает в движение.
    moving = target_angle_delta.abs() > np.deg2rad(1.0)
    required_change = 0.5 * target_angle_delta.abs()
    anti_freeze = torch.relu(required_change - pred_angle_delta.abs()).square()
    anti_freeze_loss = (anti_freeze * moving).sum() / moving.sum().clamp_min(1)

    pred_projected_scaled = (pred - mean_t) / scale_t
    # Прямо привязываем каждую точку к размеченной позе из CSV.
    coordinate_loss = nn.functional.mse_loss(pred_projected_scaled, target_scaled)
    previous_flat = previous_scaled.reshape(previous_scaled.shape[0], 69)
    pred_coordinate_delta = sequence_deltas(
        pred_projected_scaled.reshape(pred_projected_scaled.shape[0], pred_projected_scaled.shape[1], 69),
        previous_flat,
    )
    target_coordinate_delta = sequence_deltas(
        target_scaled.reshape(target_scaled.shape[0], target_scaled.shape[1], 69),
        previous_flat,
    )
    coordinate_change_loss = nn.functional.mse_loss(
        pred_coordinate_delta, target_coordinate_delta
    )
    pred_center = pred_projected_scaled.mean(dim=-2)
    target_center = target_scaled.mean(dim=-2)
    previous_center = previous_scaled.mean(dim=-2)
    pred_center_delta = sequence_deltas(pred_center, previous_center)
    target_center_delta = sequence_deltas(target_center, previous_center)
    root_motion_loss = nn.functional.mse_loss(pred_center_delta, target_center_delta)

    total = (
        LEG_LENGTH_LOSS_WEIGHT * length_loss
        + LEG_ANGLE_LOSS_WEIGHT * angle_loss
        + LEG_BEND_DIRECTION_LOSS_WEIGHT * bend_direction_loss
        + MIN_LEG_ANGLE_LOSS_WEIGHT * min_angle_loss
        + BODY_ANGLE_LIMIT_LOSS_WEIGHT * body_limit_loss
        + INTER_LEG_ANGLE_LOSS_WEIGHT * inter_leg_angle_loss
        + LEG_BODY_DISTANCE_LOSS_WEIGHT * root_distance_loss
        + ANGLE_CHANGE_LOSS_WEIGHT * angle_change_loss
        + ANTI_FREEZE_LOSS_WEIGHT * anti_freeze_loss
        + ROOT_MOTION_LOSS_WEIGHT * root_motion_loss
        + COORDINATE_LOSS_WEIGHT * coordinate_loss
        + COORDINATE_CHANGE_LOSS_WEIGHT * coordinate_change_loss
    )
    return (
        total,
        angle_loss,
        angle_change_loss,
        coordinate_loss,
        coordinate_change_loss,
        pred_projected_scaled,
    )


class MotionGRU(nn.Module):
    def __init__(
        self,
        input_size=69,
        hidden_size=128,
        num_layers=NUM_LAYERS,
        dropout=0.1,
        prediction_steps=PREDICTION_STEPS,
    ):
        super().__init__()
        self.prediction_steps = prediction_steps

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.body_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, prediction_steps * len(BODY_POINTS) * 3),
        )
        # Одна общая модель для всех ног; embedding задаёт номер ноги.
        self.leg_embedding = nn.Embedding(len(LEG_POINTS), 16)
        self.leg_head = nn.Sequential(
            nn.Linear(hidden_size + 16, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, prediction_steps * 3 * 3),
        )

    def forward(self, x):
        # x: (batch, n_steps, 69)
        output, _ = self.gru(x)
        last_hidden = output[:, -1, :]
        last_frame = x[:, -1, :].reshape(-1, 23, 3)
        batch = x.shape[0]
        body_delta = self.body_head(last_hidden).reshape(
            batch, self.prediction_steps, len(BODY_POINTS), 3
        )
        leg_ids = torch.arange(len(LEG_POINTS), device=x.device)
        embeddings = self.leg_embedding(leg_ids).unsqueeze(0).expand(batch, -1, -1)
        contexts = torch.cat(
            (last_hidden.unsqueeze(1).expand(-1, len(LEG_POINTS), -1), embeddings),
            dim=-1,
        )
        leg_delta = self.leg_head(contexts).reshape(
            batch, len(LEG_POINTS), self.prediction_steps, 3, 3
        )

        prediction = last_frame.unsqueeze(1).expand(
            -1, self.prediction_steps, -1, -1
        ).clone()
        body_indices = torch.tensor(
            [point - 1 for point in BODY_POINTS], device=x.device
        )
        prediction[:, :, body_indices, :] = (
            last_frame[:, None, body_indices, :] + body_delta
        )
        for leg_index, chain in enumerate(LEG_POINTS):
            indices = torch.tensor([point - 1 for point in chain], device=x.device)
            prediction[:, :, indices, :] = (
                last_frame[:, None, indices, :] + leg_delta[:, leg_index]
            )
        return prediction


def main():
    data, columns = load_data(CSV_PATH)
    num_frames = len(data)

    print(f"Кадров: {num_frames}")
    print(f"Точек в кадре: {data.shape[1]}")
    print(f"Координат на кадр: {data.shape[1] * data.shape[2]}")
    missing_points = sorted({
        point for chain in LEG_POINTS for point in chain if point > data.shape[1]
    })
    if missing_points:
        print(
            f"Внимание: в CSV отсутствуют точки {missing_points}; "
            "цепочки с ними не войдут в угловой лосс."
        )

    if num_frames < N_STEPS + PREDICTION_STEPS + 2:
        raise ValueError(
            f"Недостаточно кадров: {num_frames}. "
            f"Для N_STEPS={N_STEPS} и PREDICTION_STEPS={PREDICTION_STEPS} "
            "нужно больше кадров."
        )

    # Хронологическое разделение исходных кадров.
    split_frame = int(num_frames * TRAIN_RATIO)

    if split_frame < N_STEPS + PREDICTION_STEPS:
        raise ValueError(
            "Слишком мало кадров для обучающей выборки. "
            "Уменьши N_STEPS/PREDICTION_STEPS или измени TRAIN_RATIO."
        )

    scaler = StandardScaler()
    scaler.fit(data[:split_frame].reshape(-1, 3))

    scaled = scaler.transform(
        data.reshape(-1, 3)
    ).reshape(num_frames, 23, 3).astype(np.float32)

    flat = scaled.reshape(num_frames, 69)

    X, y = make_windows(flat, N_STEPS, PREDICTION_STEPS)

    target_starts = np.arange(N_STEPS, num_frames - PREDICTION_STEPS + 1)
    train_mask = target_starts + PREDICTION_STEPS <= split_frame
    val_mask = target_starts >= split_frame

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError(
            "Пустая обучающая или проверочная выборка. "
            "Измени N_STEPS или TRAIN_RATIO."
        )

    print(f"Обучающих окон: {len(X_train)}")
    print(f"Проверочных окон: {len(X_val)}")

    X_train_t = torch.tensor(X_train, device=DEVICE)
    y_train_t = torch.tensor(y_train, device=DEVICE)
    X_val_t = torch.tensor(X_val, device=DEVICE)
    y_val_t = torch.tensor(y_val, device=DEVICE)

    # Тензоры для перевода предсказаний обратно в исходные координаты
    # непосредственно внутри углового лосса.
    mean_t = torch.tensor(scaler.mean_, dtype=torch.float32, device=DEVICE).view(1, 1, 3)
    scale_t = torch.tensor(scaler.scale_, dtype=torch.float32, device=DEVICE).view(1, 1, 3)
    leg_lengths = np.asarray([
        [
            np.median(np.linalg.norm(data[:split_frame, chain[1] - 1] - data[:split_frame, chain[0] - 1], axis=-1)),
            np.median(np.linalg.norm(data[:split_frame, chain[2] - 1] - data[:split_frame, chain[1] - 1], axis=-1)),
        ]
        for chain in LEG_POINTS
    ], dtype=np.float32)
    leg_lengths_t = torch.tensor(leg_lengths, dtype=torch.float32, device=DEVICE)

    # Широкий коридор для расстояния от корня каждой ноги до центральной pt1.
    body_center = data[:split_frame, BODY_CENTER_POINT - 1]
    root_distance_bounds = []
    for chain in LEG_POINTS:
        distances = np.linalg.norm(
            data[:split_frame, chain[0] - 1] - body_center, axis=-1
        )
        padding = LEG_ROOT_DISTANCE_MARGIN * float(np.median(distances))
        root_distance_bounds.append((
            max(0.0, float(np.percentile(distances, 1)) - padding),
            float(np.percentile(distances, 99)) + padding,
        ))
    root_distance_bounds = np.asarray(root_distance_bounds, dtype=np.float32)
    root_distance_bounds_t = torch.tensor(
        root_distance_bounds, dtype=torch.float32, device=DEVICE
    )
    print("Длины звеньев ног (по обучающим данным):", np.round(leg_lengths, 2).tolist())
    print("Широкие диапазоны корень—pt1:", np.round(root_distance_bounds, 2).tolist())

    model = MotionGRU(
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        prediction_steps=PREDICTION_STEPS,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE
    )
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        permutation = torch.randperm(len(X_train_t), device=DEVICE)
        train_sum = 0.0

        for start in range(0, len(permutation), BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]

            pred = model(X_train_t[idx])
            target = y_train_t[idx].reshape(-1, PREDICTION_STEPS, 23, 3)
            previous = X_train_t[idx, -1].reshape(-1, 23, 3)
            loss, _, _, _, _, _ = motion_loss(
                pred, target, previous, mean_t, scale_t,
                leg_lengths_t, root_distance_bounds_t,
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_sum += loss.item() * len(idx)

        train_loss = train_sum / len(X_train_t)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_target = y_val_t.reshape(-1, PREDICTION_STEPS, 23, 3)
            val_previous = X_val_t[:, -1].reshape(-1, 23, 3)
            (
                val_loss,
                val_angle_loss,
                val_change_loss,
                val_coordinate_loss,
                val_coordinate_change_loss,
                val_projected,
            ) = motion_loss(
                val_pred, val_target, val_previous, mean_t, scale_t,
                leg_lengths_t, root_distance_bounds_t,
            )
            val_loss = val_loss.item()
            val_angle_degrees = np.degrees(val_angle_loss.item() ** 0.5)
            val_change_degrees = np.degrees(val_change_loss.item() ** 0.5)
            val_coordinate_rmse = np.sqrt(val_coordinate_loss.item())
            val_coordinate_change_rmse = np.sqrt(val_coordinate_change_loss.item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Эпоха {epoch:03d} | "
                f"train loss: {train_loss:.6f} | "
                f"val loss: {val_loss:.6f} | "
                f"val angle RMSE: {val_angle_degrees:.3f}° | "
                f"val angle-change RMSE: {val_change_degrees:.3f}° | "
                f"val coordinate RMSE (scaled): {val_coordinate_rmse:.4f} | "
                f"val point-motion RMSE (scaled): {val_coordinate_change_rmse:.4f}"
            )

        if patience_counter >= PATIENCE:
            print(f"Остановка на эпохе {epoch}: нет улучшения.")
            break

    model.load_state_dict(best_state)
    model.eval()

    # Оценка по углам на лучшем состоянии модели.
    with torch.no_grad():
        pred_val_scaled = model(X_val_t)
        _, _, _, _, _, projected_val = motion_loss(
            pred_val_scaled,
            y_val_t.reshape(-1, PREDICTION_STEPS, 23, 3),
            X_val_t[:, -1].reshape(-1, 23, 3),
            mean_t,
            scale_t,
            leg_lengths_t,
            root_distance_bounds_t,
        )
        pred_angles = joint_angles(projected_val * scale_t + mean_t)
        true_angles = joint_angles(y_val_t.reshape(-1, PREDICTION_STEPS, 23, 3) * scale_t + mean_t)
        previous_angles = joint_angles(X_val_t[:, -1].reshape(-1, 23, 3) * scale_t + mean_t)
        baseline_angles = previous_angles.unsqueeze(1).expand_as(true_angles)
        angle_rmse = torch.mean((pred_angles - true_angles) ** 2).sqrt().item()
        baseline_angle_rmse = torch.mean((baseline_angles - true_angles) ** 2).sqrt().item()
        coordinate_rmse = torch.mean(
            ((projected_val - y_val_t.reshape(-1, PREDICTION_STEPS, 23, 3)) * scale_t) ** 2
        ).sqrt().item()

    print("\n=== Проверочная выборка ===")
    print(f"GRU на горизонте {PREDICTION_STEPS}: угол RMSE {np.degrees(angle_rmse):.3f}°")
    print(f"Последняя поза на том же горизонте: угол RMSE {np.degrees(baseline_angle_rmse):.3f}°")
    print(f"GRU: RMSE координат {coordinate_rmse:.4f} в единицах исходного CSV")

    # Предсказание следующего кадра по последним N_STEPS кадрам.
    last_window = flat[-N_STEPS:]
    x_last = torch.tensor(
        last_window[None, :, :], device=DEVICE
    )

    with torch.no_grad():
        next_scaled = model(x_last)
        next_world = project_leg_kinematics(
            next_scaled * scale_t + mean_t,
            leg_lengths_t,
            root_distance_bounds_t,
        ).cpu().numpy()[0]
    next_coords = next_world.reshape(-1, 3)

    output_df = pd.DataFrame(
        next_coords.reshape(PREDICTION_STEPS, 69),
        columns=columns,
    )
    output_df.to_csv(OUTPUT_PATH, index=False)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_steps": N_STEPS,
            "prediction_steps": PREDICTION_STEPS,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "leg_lengths": leg_lengths,
            "root_distance_bounds": root_distance_bounds,
            "columns": columns,
            "body_points": BODY_POINTS,
            "leg_points": LEG_POINTS,
            "loss": "fixed_leg_lengths_signed_bend_min_angles_body_angle_bounds_interleg_angles_root_distance_and_run_motion",
            "model_architecture": "shared_temporal_encoder_body_head_shared_leg_head_with_leg_identity_embedding",
        },
        MODEL_PATH,
    )

    print(f"\nПредсказание сохранено: {OUTPUT_PATH}")
    print(f"Модель сохранена: {MODEL_PATH}")
    print(f"\nПредсказано кадров: {PREDICTION_STEPS}")
    print("Первые 3 точки первого предсказанного кадра:")
    print(next_coords[:3])


if __name__ == "__main__":
    main()
