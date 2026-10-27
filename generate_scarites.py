
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from main import MotionGRU, project_leg_kinematics


# ==================== НАСТРОЙКИ ====================

CSV_PATH = "data/Scarites_6_smoothed_coords.csv"
MODEL_PATH = "scarites_gru.pt"
# The Geometry Nodes file is in data/ and imports this sibling CSV by name.
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "Scarites_6_generated_1100.csv"


NUM_GENERATED_FRAMES = 1000

DEVICE = torch.device("cpu")

# ==================================================


def main():
    # 1. Загружаем исходную последовательность.
    df = pd.read_csv(CSV_PATH, sep=None, engine="python")
    df.columns = df.columns.astype(str).str.strip()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    required_keys = ("prediction_steps", "leg_lengths")
    missing_checkpoint_keys = [key for key in required_keys if key not in checkpoint]
    if missing_checkpoint_keys:
        raise ValueError(
            "Этот checkpoint создан старой одношаговой моделью. "
            "Сначала переобучи main.py, чтобы сохранить прогноз M кадров."
        )

    columns = checkpoint["columns"]

    # Проверяем, что формат исходного CSV соответствует модели.
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"В исходном CSV отсутствуют столбцы: {missing}")

    data = df[columns].to_numpy(dtype=np.float32)
    data = data.reshape(-1, 23, 3)

    # 2. Восстанавливаем параметры масштабирования.
    mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)

    if np.any(scale == 0):
        raise ValueError("Некорректный масштаб координат.")

    n_steps = int(checkpoint["n_steps"])
    prediction_steps = int(checkpoint["prediction_steps"])
    leg_lengths = torch.tensor(
        checkpoint["leg_lengths"], dtype=torch.float32, device=DEVICE
    )
    root_distance_bounds = torch.tensor(
        checkpoint["root_distance_bounds"], dtype=torch.float32, device=DEVICE
    )

    if len(data) < n_steps:
        raise ValueError(
            f"Для истории модели нужно {n_steps} кадров, "
            f"а в CSV найдено {len(data)}."
        )

    # 3. Восстанавливаем архитектуру и загружаем веса.
    model = MotionGRU(
        input_size=69,
        hidden_size=int(checkpoint["hidden_size"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
        prediction_steps=prediction_steps,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 4. Начинаем с настоящей истории из первых N кадров.
    history = ((data[:n_steps] - mean) / scale).reshape(n_steps, 69)
    generated_frames = [frame.copy() for frame in data[:n_steps]]

    # 5. Авторегрессионная генерация.
    with torch.no_grad():
        generated_count = 0
        while generated_count < NUM_GENERATED_FRAMES:
            x = torch.tensor(
                history[None, :, :],
                dtype=torch.float32,
                device=DEVICE,
            )

            # Модель сразу предсказывает M кадров; проекция держит длины,
            # минимум угла 30° и широкую дистанцию корней от центра тела.
            next_scaled_t = model(x)[0]
            next_world_t = torch.tensor(
                next_scaled_t.cpu().numpy() * scale + mean,
                dtype=torch.float32,
                device=DEVICE,
            )
            next_world = project_leg_kinematics(
                next_world_t, leg_lengths, root_distance_bounds
            ).cpu().numpy()
            next_scaled = (next_world - mean) / scale

            remaining = NUM_GENERATED_FRAMES - generated_count
            frames_to_add = min(prediction_steps, remaining)
            generated_frames.extend(frame.copy() for frame in next_world[:frames_to_add])
            generated_count += frames_to_add

            # В следующее окно попадают уже спроецированные кадры.
            next_flat = next_scaled.reshape(prediction_steps, 69)
            history = np.concatenate(
                [history, next_flat], axis=0
            )[-n_steps:]

            if generated_count % 100 < prediction_steps or generated_count == NUM_GENERATED_FRAMES:
                print(
                    f"Сгенерировано кадров: "
                    f"{generated_count}/{NUM_GENERATED_FRAMES}"
                )

    # 6. Сохраняем исходную историю + новые кадры.
    result = np.asarray(generated_frames).reshape(-1, 69)

    output_df = pd.DataFrame(result, columns=columns)
    output_df.to_csv(OUTPUT_PATH, index=False, float_format="%.8f")

    print(f"\nГотово: {Path(OUTPUT_PATH).resolve()}")
    print(f"Всего кадров: {len(output_df)}")
    print(f"Начальная история: первые {n_steps} исходных кадров")
    print(f"Сгенерировано новых кадров: {NUM_GENERATED_FRAMES}")


if __name__ == "__main__":
    main()
