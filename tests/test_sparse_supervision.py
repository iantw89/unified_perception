import math

import numpy as np
import torch
import torch.nn.functional as F

from unified_perception.config import Config, ModelConfig, load_config
from unified_perception.constants import NUM_JOINTS, ROOT_JOINTS
from unified_perception.data.collate import collate_batch
from unified_perception.data.backends import _build_output_window
from unified_perception.engine.metrics import MetricsAccumulator
from unified_perception.engine.evaluate import run_eval
from unified_perception.models.losses import SetCriterion
from unified_perception.models.matcher import hungarian_match


def test_matcher_switches_from_keypoints_to_root_on_unlabeled_frames():
    logits = torch.zeros(2, 1)
    pred = torch.stack([torch.full((NUM_JOINTS, 3), x) for x in (0.1, 0.8)])
    gt = torch.stack([torch.full((NUM_JOINTS, 3), x) for x in (0.1, 0.8)])
    roots = torch.tensor([[0.8, 0.8, 0.8], [0.1, 0.1, 0.1]])

    _, dense_gt = hungarian_match(logits, pred, gt, cost_class=0, cost_kpt=1)
    _, sparse_gt = hungarian_match(
        logits, pred, gt, cost_class=0, cost_kpt=1,
        gt_root=roots, cost_root=1, root_only=True,
    )

    assert dense_gt.tolist() == [0, 1]
    assert sparse_gt.tolist() == [1, 0]


def test_zero_supervised_frames_train_cls_and_root_only():
    loss_cfg = Config().loss
    loss_cfg.root_weight = 1.0
    criterion = SetCriterion(ModelConfig(), loss_cfg)
    logits = torch.zeros(1, 1, 1, 2, 1, requires_grad=True)
    keypoints = torch.full((1, 1, 1, 2, NUM_JOINTS, 3), 0.5, requires_grad=True)
    gt = torch.zeros(1, 1, 1, NUM_JOINTS, 3)
    valid = torch.ones(1, 1, 1, dtype=torch.bool)
    root = torch.full((1, 1, 1, 3), 0.2)
    frame_valid = torch.zeros(1, 1, dtype=torch.bool)

    loss, metrics = criterion(
        logits, keypoints, gt, valid,
        gt_keypoint_frame_valid=frame_valid,
        gt_root=root,
        gt_root_valid=valid,
    )
    loss.backward()

    assert math.isfinite(float(loss.detach()))
    assert metrics["loss_kpt"] == 0
    assert metrics["num_kpt"] == 0
    assert metrics["loss_root"] > 0
    assert logits.grad is not None and logits.grad.abs().sum() > 0
    assert keypoints.grad is not None
    non_root = [j for j in range(NUM_JOINTS) if j not in ROOT_JOINTS]
    assert torch.count_nonzero(keypoints.grad[..., non_root, :]) == 0
    assert torch.count_nonzero(keypoints.grad[..., list(ROOT_JOINTS), :]) > 0


def test_dense_defaults_keep_previous_loss_formula():
    model_cfg = ModelConfig()
    criterion = SetCriterion(model_cfg)
    logits = torch.tensor([[[[[0.2], [-0.4]]]]])
    pred = torch.full((1, 1, 1, 2, NUM_JOINTS, 3), 0.7)
    pred[:, :, :, 0] = 0.2
    gt = torch.full((1, 1, 1, NUM_JOINTS, 3), 0.2)
    valid = torch.ones(1, 1, 1, dtype=torch.bool)

    loss, metrics = criterion(logits, pred, gt, valid)
    target = torch.tensor([[[[[1.0], [0.0]]]]])
    expected_cls = model_cfg.loss_class * F.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    prob = logits.sigmoid()
    focal = torch.where(target.bool(), 0.25 * (1 - prob) ** 2, 0.75 * prob ** 2) * expected_cls
    expected_cls = focal.sum()
    expected_kpt = model_cfg.loss_kpt * F.l1_loss(pred[0, 0, 0, 0], gt[0, 0, 0], reduction="sum") / (NUM_JOINTS * 3)

    assert torch.allclose(loss, expected_cls + expected_kpt)
    assert math.isclose(metrics["loss_root"], 0.0)


def test_collate_builds_dense_and_derived_root_defaults():
    keypoints = torch.arange(2 * 1 * NUM_JOINTS * 3, dtype=torch.float32).reshape(2, 1, NUM_JOINTS, 3)
    valid = torch.tensor([[True], [False]])
    sample = {
        "imgs": torch.zeros(2, 1, 3, 4, 4, dtype=torch.uint8),
        "world2img": torch.zeros(2, 1, 4, 4),
        "intrinsics": torch.zeros(2, 1, 4, 4),
        "gt_keypoints": keypoints,
        "gt_valid": valid,
        "seq": "dense",
        "start": 0,
    }

    batch = collate_batch([sample])

    assert batch["gt_keypoint_frame_valid"].tolist() == [[True, True]]
    assert batch["gt_keypoint_valid"].tolist() == [[[[True] * NUM_JOINTS], [[False] * NUM_JOINTS]]]
    assert batch["gt_root_valid"].tolist() == [[[True], [False]]]
    expected = keypoints[..., list(ROOT_JOINTS), :].mean(dim=-2)
    assert torch.equal(batch["gt_root"][0], expected)


def test_sparse_metrics_count_and_root_on_unlabeled_frames_only():
    metrics = MetricsAccumulator(match_threshold_mm=100.0)
    gt_kpts = np.zeros((1, NUM_JOINTS, 3), dtype=np.float32)
    pred_kpts = np.full_like(gt_kpts, 1000.0)
    root = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)

    metrics.update_frame(
        pred_kpts, gt_kpts, keypoints_supervised=False,
        pred_root_mm=root, gt_root_mm=root.copy(), gt_root_valid=np.array([True]),
    )
    result = metrics.compute()

    assert result["target_persons"] == 1
    assert result["precision"] == 1.0 and result["recall"] == 1.0
    assert math.isnan(result["mpjpe_mm"])
    assert result["root_error_mm"] == 0.0


def test_output_window_preserves_optional_sparse_targets():
    keypoints = np.zeros((2, 1, NUM_JOINTS, 3), dtype=np.float32)
    keypoints[:, 0, list(ROOT_JOINTS)] = 25.0
    valid = np.ones((2, 1), dtype=bool)
    roots = np.full((2, 1, 3), 123.0, dtype=np.float32)
    roots_valid = np.array([[True], [False]])
    identity = np.eye(3, dtype=np.float64)

    out = _build_output_window(
        seq="s", start=0, seq_len=2, out_hw=(2, 2), max_people=1, num_cameras=1,
        keypoints=keypoints, valid=valid, K_by_cam=identity[None],
        R_by_cam=identity[None], T_by_cam=np.zeros((1, 3)),
        image_loader=lambda _t, _c, K: (torch.zeros(3, 2, 2, dtype=torch.uint8), K),
        keypoint_frame_valid=np.array([True, False]),
        gt_root=roots, gt_root_valid=roots_valid,
    )

    assert out["gt_keypoint_frame_valid"].tolist() == [True, False]
    assert np.array_equal(out["gt_root"], roots)
    assert np.array_equal(out["gt_root_valid"], roots_valid)


def test_sparse_config_keys_are_loadable_and_backward_compatible():
    cfg = load_config(overrides=["loss.root_weight=5.0", "model.cost_root=3.0"])
    assert cfg.loss.root_weight == 5.0
    assert cfg.model.cost_root == 3.0
    assert Config().loss.root_weight == 0.0


def test_eval_keeps_detection_and_root_metrics_on_unlabeled_frames():
    class DummyModel(torch.nn.Module):
        def forward(self, imgs, _world2img, _intrinsics, camera_valid=None):
            batch, frames = imgs.shape[:2]
            scores = torch.full((1, batch, frames, 1, 1), 10.0, device=imgs.device)
            pred = torch.zeros((1, batch, frames, 1, NUM_JOINTS, 3), device=imgs.device)
            return scores, pred, None, None

        def denormalize_keypoints(self, keypoints):
            return keypoints

    cfg = Config()
    batch = {
        "imgs": torch.zeros(1, 2, 1, 3, 2, 2, dtype=torch.uint8),
        "world2img": torch.zeros(1, 2, 1, 4, 4),
        "intrinsics": torch.zeros(1, 2, 1, 4, 4),
        "gt_keypoints": torch.zeros(1, 2, 1, NUM_JOINTS, 3),
        "gt_valid": torch.ones(1, 2, 1, dtype=torch.bool),
        "gt_keypoint_frame_valid": torch.tensor([[True, False]]),
        "gt_root": torch.zeros(1, 2, 1, 3),
        "gt_root_valid": torch.ones(1, 2, 1, dtype=torch.bool),
        "seq": ["s"], "start": [0],
    }
    batch["gt_keypoints"][0, 1] = 1000.0

    result = run_eval(DummyModel(), [batch], cfg, torch.device("cpu"))

    assert result["target_persons"] == 2
    assert result["matched_persons"] == 2
    assert result["count_mae"] == 0.0
    assert result["mpjpe_mm"] == 0.0
    assert result["root_error_mm"] == 0.0


def test_matcher_uses_root_for_person_with_no_valid_joints_on_labeled_frame():
    logits = torch.zeros(2, 1)
    pred = torch.stack([torch.zeros(NUM_JOINTS, 3), torch.ones(NUM_JOINTS, 3)])
    gt = torch.zeros(1, NUM_JOINTS, 3)
    joint_valid = torch.zeros(1, NUM_JOINTS, dtype=torch.bool)
    root = torch.ones(1, 3)

    pred_idx, gt_idx = hungarian_match(
        logits, pred, gt, gt_root=root,
        gt_root_valid=torch.ones(1, dtype=torch.bool),
        gt_keypoint_valid=joint_valid, cost_root=5.0,
    )

    assert pred_idx.tolist() == [1]
    assert gt_idx.tolist() == [0]


def test_keypoint_loss_and_matching_ignore_invalid_joints():
    criterion = SetCriterion(ModelConfig())
    logits = torch.zeros(1, 1, 1, 1, 1)
    pred = torch.zeros(1, 1, 1, 1, NUM_JOINTS, 3, requires_grad=True)
    with torch.no_grad():
        pred[..., 5, :] = 0.6
    gt = torch.zeros(1, 1, 1, NUM_JOINTS, 3)
    valid = torch.ones(1, 1, 1, dtype=torch.bool)
    joint_valid = torch.zeros(1, 1, 1, NUM_JOINTS, dtype=torch.bool)
    joint_valid[..., 5] = True

    loss, metrics = criterion(
        logits, pred, gt, valid, gt_keypoint_valid=joint_valid,
    )
    loss.backward()

    assert metrics["num_kpt"] == 1
    assert torch.isclose(torch.tensor(metrics["loss_kpt"]), torch.tensor(3.0))
    assert torch.count_nonzero(pred.grad[..., :5, :]) == 0
    assert torch.count_nonzero(pred.grad[..., 6:, :]) == 0
    assert torch.count_nonzero(pred.grad[..., 5, :]) == 3


def test_dn_keypoint_loss_ignores_invalid_joints():
    criterion = SetCriterion(ModelConfig())
    dn_kpts = torch.zeros(1, 1, 1, 1, NUM_JOINTS, 3, requires_grad=True)
    with torch.no_grad():
        dn_kpts[..., 0, :] = 50.0
    dn = {
        "dn_cls": torch.zeros(1, 1, 1, 1, 1, requires_grad=True),
        "dn_kpts": dn_kpts,
        "target_kpts": torch.zeros(1, 1, 1, NUM_JOINTS, 3),
        "target_kpt_valid": torch.zeros(1, 1, 1, NUM_JOINTS, dtype=torch.bool),
        "valid": torch.ones(1, 1, 1, dtype=torch.bool),
        "is_person": torch.ones(1, 1, 1, dtype=torch.bool),
    }
    dn["target_kpt_valid"][..., 4] = True

    _, loss_kpt = criterion._dn_loss(dn)

    assert loss_kpt == 0


def test_metrics_use_valid_joint_subset_and_root_fallback():
    metrics = MetricsAccumulator(pck_thresholds_mm=[5.0])
    gt = np.zeros((1, NUM_JOINTS, 3), dtype=np.float32)
    pred = np.zeros_like(gt)
    gt[0, 0] = 100.0
    mask = np.zeros((1, NUM_JOINTS), dtype=bool)
    mask[0, 4] = True
    zero_root = np.zeros((1, 3), dtype=np.float32)

    metrics.update_frame(
        pred, gt, keypoints_supervised=True,
        pred_root_mm=zero_root, gt_root_mm=zero_root,
        gt_keypoint_valid=mask,
    )
    result = metrics.compute()
    assert result["mpjpe_mm"] == 0.0
    assert result["pck@5mm"] == 1.0

    root_only = MetricsAccumulator()
    far_pose = np.full_like(gt, 1000.0)
    root_only.update_frame(
        far_pose, gt, keypoints_supervised=True,
        pred_root_mm=zero_root, gt_root_mm=zero_root,
        gt_keypoint_valid=np.zeros_like(mask),
        gt_root_valid=np.array([True]),
    )
    result = root_only.compute()
    assert result["precision"] == 1.0 and result["recall"] == 1.0
    assert math.isnan(result["mpjpe_mm"])
