"""Skeleton / keypoint definitions, shared across data and model.

A fixed 12-joint skeleton (matching the human_prior project) is used so that
mocap ground truth, rendering, and metrics all agree on joint ordering.
"""

from __future__ import annotations

# Canonical joint order (index -> name).
JOINT_NAMES = [
    "left_hip",       # 0
    "right_hip",      # 1
    "left_shoulder",  # 2
    "right_shoulder", # 3
    "left_elbow",     # 4
    "right_elbow",    # 5
    "left_wrist",     # 6
    "right_wrist",    # 7
    "left_knee",      # 8
    "right_knee",     # 9
    "left_ankle",     # 10
    "right_ankle",    # 11
]

NUM_JOINTS = len(JOINT_NAMES)

JOINT_INDEX = {name: i for i, name in enumerate(JOINT_NAMES)}

# Bones for rendering / visualisation (pairs of joint indices).
SKELETON_EDGES = [
    (JOINT_INDEX["left_hip"], JOINT_INDEX["right_hip"]),
    (JOINT_INDEX["left_shoulder"], JOINT_INDEX["right_shoulder"]),
    (JOINT_INDEX["left_hip"], JOINT_INDEX["left_shoulder"]),
    (JOINT_INDEX["right_hip"], JOINT_INDEX["right_shoulder"]),
    (JOINT_INDEX["left_shoulder"], JOINT_INDEX["left_elbow"]),
    (JOINT_INDEX["left_elbow"], JOINT_INDEX["left_wrist"]),
    (JOINT_INDEX["right_shoulder"], JOINT_INDEX["right_elbow"]),
    (JOINT_INDEX["right_elbow"], JOINT_INDEX["right_wrist"]),
    (JOINT_INDEX["left_hip"], JOINT_INDEX["left_knee"]),
    (JOINT_INDEX["left_knee"], JOINT_INDEX["left_ankle"]),
    (JOINT_INDEX["right_hip"], JOINT_INDEX["right_knee"]),
    (JOINT_INDEX["right_knee"], JOINT_INDEX["right_ankle"]),
]

# Distinct colours (RGB) per joint for rendering skeletons into images.
JOINT_COLORS = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
]

# The "root" joint used as a person's reference point (mid-hip is derived from
# left/right hip; we use left_hip index for simple per-query reference anchoring).
ROOT_JOINTS = (JOINT_INDEX["left_hip"], JOINT_INDEX["right_hip"])
