"""Generate Scarites_6 motion with the seven independently trained models."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from multi_beetle import (
    BODY_CENTER,
    DEVICE,
    GROUPS,
    IndependentGroupGRU,
    N_STEPS,
    PREDICTION_STEPS,
    ROOT,
    canonical_frame,
    expected_columns,
    load_subject,
    limit_leg_frame_change,
    project_full_pose,
    subject_geometry,
)


SOURCE_PATH = ROOT / "data" / "Scarites_6_smoothed_coords.csv"
CHECKPOINT_PATH = ROOT / "multi_beetle_gru.pt"
OUTPUT_PATH = ROOT / "data" / "Scarites_6_generated_1000.csv"
NUM_GENERATED_FRAMES = 1000


def main():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    source, columns = load_subject(SOURCE_PATH)
    if columns != expected_columns():
        raise ValueError("Source CSV columns do not match the trained model.")
    if len(source) < N_STEPS:
        raise ValueError(f"Need {N_STEPS} seed frames; found {len(source)}.")

    canonical, origin, basis = canonical_frame(source)
    mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
    scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
    if np.any(scale == 0):
        raise ValueError("Invalid saved coordinate scale.")
    scaled = ((canonical - mean) / scale).astype(np.float32)
    history = scaled[:N_STEPS].reshape(N_STEPS, 69)
    generated = [frame.copy() for frame in source[:N_STEPS]]

    models = {}
    for name, point_ids in GROUPS.items():
        model = IndependentGroupGRU(
            [point - 1 for point in point_ids],
            hidden_size=int(checkpoint["hidden_size"]),
            num_layers=int(checkpoint["num_layers"]),
            dropout=float(checkpoint["dropout"]),
        ).to(DEVICE)
        model.load_state_dict(checkpoint["model_states"][name])
        model.eval()
        models[name] = model

    train_end = int(len(source) * 0.8)
    geometry = subject_geometry(canonical, train_end)
    lengths = torch.tensor(geometry["leg_lengths"], dtype=torch.float32, device=DEVICE)
    bounds = torch.tensor(geometry["root_distance_bounds"], dtype=torch.float32, device=DEVICE)
    body_edge_lengths = torch.tensor(
        geometry["body_edge_lengths"], dtype=torch.float32, device=DEVICE
    )
    caps = checkpoint["motion_caps"]
    mean_t = torch.tensor(mean, dtype=torch.float32, device=DEVICE).view(1, 1, 3)
    scale_t = torch.tensor(scale, dtype=torch.float32, device=DEVICE).view(1, 1, 3)

    generated_count = 0
    with torch.no_grad():
        while generated_count < NUM_GENERATED_FRAMES:
            x = torch.tensor(history[None], dtype=torch.float32, device=DEVICE)
            next_scaled = x[:, -1].reshape(1, 23, 3).expand(
                1, PREDICTION_STEPS, 23, 3
            ).clone()[0]
            for name, point_ids in GROUPS.items():
                prediction = models[name](x)[0]
                indices = torch.tensor([p - 1 for p in point_ids], device=DEVICE)
                next_scaled[:, indices, :] = prediction

            next_canonical = next_scaled * scale_t + mean_t
            next_canonical = project_full_pose(
                next_canonical, lengths, bounds, body_edge_lengths
            )
            previous_pose = (
                torch.tensor(history[-1].reshape(23, 3), dtype=torch.float32, device=DEVICE)
                * scale_t[0, 0] + mean_t[0, 0]
            )
            limited_frames = []
            for frame in next_canonical:
                frame = limit_leg_frame_change(
                    previous_pose,
                    frame,
                    lengths,
                    caps["max_angle_step"],
                    caps["max_bend_normal_turn"],
                )
                limited_frames.append(frame)
                previous_pose = frame
            next_canonical = torch.stack(limited_frames, dim=0)
            next_scaled = (next_canonical - mean_t) / scale_t
            next_world = next_canonical.cpu().numpy() @ basis.T + origin

            count = min(PREDICTION_STEPS, NUM_GENERATED_FRAMES - generated_count)
            generated.extend(frame.copy() for frame in next_world[:count])
            generated_count += count
            history = np.concatenate(
                (history, next_scaled.cpu().numpy().reshape(PREDICTION_STEPS, 69)),
                axis=0,
            )[-N_STEPS:]
            if generated_count % 100 < PREDICTION_STEPS or generated_count == NUM_GENERATED_FRAMES:
                print(f"Generated {generated_count}/{NUM_GENERATED_FRAMES} frames")

    output = pd.DataFrame(np.asarray(generated).reshape(-1, 69), columns=columns)
    output.to_csv(OUTPUT_PATH, index=False, float_format="%.8f")
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(output)} (seed {N_STEPS} + generated {NUM_GENERATED_FRAMES})")


if __name__ == "__main__":
    main()
