"""Shared paths and tunable parameters for stages 0 through 3."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"                         # source recordings only
TRANSLATE_DIR = ROOT / "translate"               # stage 0 features and morphology
MODEL_DIR = ROOT / "models"                     # stage 1 checkpoints
GENERATION_DIR = ROOT / "generation"             # stage 2 velocity predictions
OUTPUT_DOTS_DIR = ROOT / "output_dots"           # stage 3 point-coordinate CSVs
ANALYSIS_DIR = ROOT / "analysis"                 # per-recording diagnostic reports

DEFAULT_STAGE = 0
TARGET_NAME = "Scarites_6"
SOURCE_GLOB = "*_smoothed_coords.csv"
TRAIN_FRACTION = 0.8
# 1-based indices in sorted data/*_smoothed_coords.csv order. Use (1, 2, 3, 4)
# for all data; e.g. (1, 3) creates the training prefix f13.
TRAIN_FILE_INDICES = (1, 2, 3, 4)

# Temporal model parameters.
HISTORY_N = 50
PREDICTION_M = 7
GENERATED_STEPS = 1000
def source_dataset_paths():
    return tuple(sorted(DATA_DIR.glob(SOURCE_GLOB)))


def selected_training_paths():
    paths = source_dataset_paths()
    invalid = [index for index in TRAIN_FILE_INDICES if index < 1 or index > len(paths)]
    if invalid:
        raise ValueError(f"Training file indices out of range: {invalid}; found {len(paths)} files")
    if len(set(TRAIN_FILE_INDICES)) != len(TRAIN_FILE_INDICES):
        raise ValueError("TRAIN_FILE_INDICES cannot contain duplicates")
    return tuple(paths[index - 1] for index in TRAIN_FILE_INDICES)


def training_prefix():
    selected = tuple(sorted(TRAIN_FILE_INDICES))
    if selected == tuple(range(1, len(source_dataset_paths()) + 1)):
        return "all"
    return "f" + "".join(str(index) for index in selected)


TRAIN_PREFIX = training_prefix()
RUN_PARAMETER_TAG = f"{TRAIN_PREFIX}_N{HISTORY_N}_M{PREDICTION_M}_F{GENERATED_STEPS}"
BODY_MODEL_PATH = MODEL_DIR / f"body_{RUN_PARAMETER_TAG}.pt"
LEGS_MODEL_PATH = MODEL_DIR / f"legs_{RUN_PARAMETER_TAG}.pt"
HIDDEN_SIZE = 128
GRU_LAYERS = 2
DROPOUT = 0.1
EPOCHS = 80
BATCH_SIZE = 64
LEARNING_RATE = 8e-4
EARLY_STOPPING_PATIENCE = 12
RANDOM_SEED = 43

# Anatomy: body triangles share pt1; legs are listed proximal to distal.
FRONT_TRIANGLE = (1, 2, 3)
REAR_TRIANGLE = (1, 4, 5)
LEG_CHAINS = (
    (11, 17, 23),
    (9, 15, 21),
    (7, 13, 19),
    (6, 12, 18),
    (8, 14, 20),
    (10, 16, 22),
)
# Virtual hip anchors, selected from the tracked body points. Distal pairings
# follow the nearest body point in the four recordings; legs 2 and 4 use pt1.
LEG_ANCHORS = (5, 1, 3, 1, 2, 4)
JOINTS_PER_LEG = 3  # body anchor->root, root->middle, middle->tip
JOINT_COUNT = len(LEG_CHAINS) * JOINTS_PER_LEG

# Scarites_6 fallback morphology: three link lengths per leg, in point units.
DEFAULT_LEG_LENGTHS = (
    (11.70, 6.87, 4.65),
    (9.86, 4.99, 3.67),
    (12.46, 7.81, 3.92),
    (12.30, 4.86, 3.84),
    (10.02, 5.36, 3.38),
    (12.14, 5.79, 5.12),
)
DEFAULT_FRONT_TRIANGLE_EDGES = (9.30, 9.29, 9.49)  # 1-2, 1-3, 2-3
DEFAULT_REAR_TRIANGLE_EDGES = (13.27, 12.94, 5.44)  # 1-4, 1-5, 4-5

# Stage 0 estimates per-recording limits, with this margin, and writes them to
# translate/morphology.json. These fallback values are only used if absent.
ANGULAR_SPEED_MARGIN = 1.2
ANGULAR_ACCELERATION_MARGIN = 1.2
DEFAULT_MAX_ANGULAR_SPEED = 0.328  # rad/frame
DEFAULT_MAX_ANGULAR_ACCELERATION = 0.126  # rad/frame^2

BODY_CENTER_FIT_WEIGHT = 2.0
BODY_ANGULAR_FIT_WEIGHT = 20.0
LEG_ANGULAR_FIT_WEIGHT = 100.0
# All auxiliary leg penalties combined contribute at most this amount.
LEG_REGULARIZATION_MAX = 1.0
ANGULAR_ACCELERATION_CAP_WEIGHT = 100.0
ANGULAR_SPEED_CAP_WEIGHT = 50.0

# Geometry regularizers use heights measured along the source recordings' real
# vertical axis. Limits are estimated by stage 0 from the selected recordings.
BELLY_TOO_LOW_WEIGHT = 8.0
FEET_TOO_HIGH_WEIGHT = 8.0
FRONT_NORMAL_TURN_WEIGHT = 20.0
REAR_NORMAL_PARALLEL_WEIGHT = 80.0
REAR_NORMAL_PARALLEL_MAX_DEG = 30.0
MAX_FRONT_ROTATION_FROM_SEED_DEG = 15.0
MAX_REAR_ROTATION_FROM_FRONT_SEED_DEG = 30.0
VIRTUAL_HIP_ANGLE_LIMIT_WEIGHT = 20.0
# Maxima use all selected recordings; speed floors use their training split.
# The floor applies to mean speed across M frames, allowing step reversals.
LEG_MIN_SPEED_PERCENTILE = 25.0
LEG_MIN_SPEED_FRACTION = 0.5
LEG_MIN_SPEED_WEIGHT = 30.0
LEG_MAX_SPEED_WEIGHT = 100.0
ANGULAR_SPEED_EXP_STEEPNESS = 4.0
# First two links: body anchor -> leg root, then root -> middle.
LEG_GROUND_MIN_DOWNWARD_DEG = 5.0
LEG_GROUND_DIRECTION_WEIGHT = 100.0
VIRTUAL_HIP_ELEVATION_WEIGHT = 80.0
HEADING_ALIGNMENT_WEIGHT = 50.0
HEAD_GROUND_PARALLEL_WEIGHT = 100.0
HEADING_TURN_PROGRESS_WEIGHT = 2.5
HEADING_MIN_SPEED = 1e-3  # Ignore stationary frames when comparing heading and travel.

INPUT_FEATURES = 11 + JOINT_COUNT * 4
BODY_FEATURES = 11
LEG_FEATURES = JOINT_COUNT * 4

