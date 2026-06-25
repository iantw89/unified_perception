"""Unified Perception: a StreamPETR-style multi-view, multi-person 3D keypoint model.

Adapts the StreamPETR approach (multi-view 2D images + camera calibration -> a
transformer detector with temporal memory and Hungarian matching) to the
human-pose domain: predict the 3D keypoints of an arbitrary number of people
from calibrated multi-view images, supervised directly by mocap ground truth.
"""

__version__ = "0.1.0"
