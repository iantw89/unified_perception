"""Approximate-GT adaptation (data/golden.py), noisy/2-D supervision, camera dropout, eval matching."""

import cv2
import numpy as np
import torch

from unified_perception.config import ModelConfig
from unified_perception.constants import JOINT_NAMES
from unified_perception.data.augment import drop_cameras
from unified_perception.data.golden import COCO_TO_JOINTS, Camera, GoldenConfig, build_targets
from unified_perception.engine.evaluate import _match_two_d
from unified_perception.models.losses import SetCriterion
from unified_perception.models.supervision import LossSettings, Supervision, match, noisy_l1, pair_losses, project

J = len(JOINT_NAMES)


def camera(name, position, look_at=(0.0, 0.0, 40.0)):
    """Pinhole camera (canvas == video, no distortion) looking at ``look_at`` from ``position`` (inches)."""
    position, target = np.asarray(position, float), np.asarray(look_at, float)
    z = (target - position) / np.linalg.norm(target - position)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    rvec = cv2.Rodrigues(R)[0].reshape(3)
    K = np.array([[600.0, 0, 640], [0, 600.0, 480], [0, 0, 1]])
    return Camera(name, K, np.zeros(5), rvec, -R @ position, 1.0, np.zeros(2), 1.0, (960, 1280))


def body(x, y):
    """COCO-17 standing person (inches) at floor position (x, y)."""
    heights = [66, 67, 67, 66, 66, 58, 58, 46, 46, 36, 36, 38, 38, 20, 20, 3, 3]
    side = [0, -1, 1, -2, 2, -7, 7, -9, 9, -9, 9, -4, 4, -4, 4, -4, 4]
    return np.array([[x + s, y, h] for s, h in zip(side, heights)], float)


def scene():
    cams = [camera("golden-0", (-150, -150, 100)), camera("golden-1", (150, -150, 100))]
    a, b = body(0, 0), body(30, 40)
    tracks = {"units": "in", "keypoint_format": "coco17", "tracks": [
        {"track_id": 0, "members": [{"camera": "golden-0", "tracklet_id": 1, "frame_range": [0, 20]},
                                    {"camera": "golden-1", "tracklet_id": 1, "frame_range": [0, 20]}],
         "positions": [{"aligned_index": f, "xyz": list(a[11:13].mean(0) + [f, 0, 0]),
                        "keypoints": (a + [f, 0, 0]).tolist()} for f in (0, 5, 10)]},
        # Seen by camera 1 only: no 3-D positions at all.
        {"track_id": 1, "members": [{"camera": "golden-1", "tracklet_id": 2, "frame_range": [0, 20]}],
         "positions": []},
    ]}
    tracklets = {}
    for cam in cams:
        frames = []
        for f in (0, 5, 10):
            dets = []
            for tid, pts in ((1, a + [f, 0, 0]), (2, b)):
                if tid == 2 and cam.name == "golden-0":
                    continue
                uv = cam.project_canvas(pts) + 1.0  # 1 px detector noise
                dets.append({"tracklet_id": tid, "keypoints": [[u, v, 0.9] for u, v in uv]})
            frames.append({"frame": f, "detections": dets})
        tracklets[cam.name] = {"camera": cam.name, "frames": frames}
    alignment = {"checkpoints": [{"aligned_index": f, "frames": {c.name: f for c in cams}} for f in range(11)],
                 "source": {"valid_aligned_range": [0, 10]}}
    return tracks, tracklets, alignment, cams


def test_golden_targets_sigma_interpolation_and_untriangulated_people():
    tracks, tracklets, alignment, cams = scene()
    arrays = build_targets(tracks, tracklets, alignment, cams,
                           GoldenConfig(calibration_sigma_mm=10.0, interpolated_use="loss"))
    assert arrays["labeled"].nonzero()[0].tolist() == [0, 5, 10]
    assert arrays["triangulated"][[0, 5, 10], 0].all() and not arrays["triangulated"][:, 1].any()
    assert arrays["observed"][0, 1].tolist() == [False, True]          # person 1: one camera only
    assert arrays["keypoint_valid"][0, 0].all() and not arrays["keypoint_valid"][:, 1].any()
    sigma = arrays["keypoint_sigma"][0, 0]
    assert (sigma >= 10.0).all() and (sigma < 40.0).all()                # floor + small residual term
    np.testing.assert_allclose(arrays["keypoints"][0, 0], body(0, 0)[COCO_TO_JOINTS] * 25.4, atol=1e-3)
    # Frame 2 interpolates between 0 and 5 with a motion term in its sigma.
    assert arrays["interpolated"][2, 0] and arrays["keypoint_valid"][2, 0].all()
    np.testing.assert_allclose(arrays["keypoints"][2, 0, 0, 0], (body(0, 0)[COCO_TO_JOINTS][0, 0] + 2) * 25.4,
                               atol=1e-3)
    assert (arrays["keypoint_sigma"][2, 0] > sigma).all()
    assert arrays["observed"][2].tolist() == arrays["observed"][0].tolist()  # existence copies nearest label
    np.testing.assert_allclose(arrays["kp2d_video"][0, 1, 1, :, :2],
                               cams[1].project_canvas(body(30, 40))[COCO_TO_JOINTS] + 1.0, atol=1e-6)


def test_golden_match_mode_keeps_interpolated_roots_for_matching_only():
    tracks, tracklets, alignment, cams = scene()
    loss = build_targets(tracks, tracklets, alignment, cams, GoldenConfig(interpolated_use="loss"))
    tracks, tracklets, alignment, cams = scene()
    arrays = build_targets(tracks, tracklets, alignment, cams, GoldenConfig())   # default: match
    between = ~arrays["labeled"]
    assert not arrays["keypoint_valid"][between].any() and not arrays["root_valid"][between].any()
    assert not arrays["interpolated"].any()
    assert arrays["match_root_valid"][2, 0] and not arrays["match_root_valid"][arrays["labeled"]].any()
    np.testing.assert_allclose(arrays["match_root"][2, 0], loss["root"][2, 0], atol=1e-3)
    labeled = arrays["labeled"]                                     # label frames are unchanged
    np.testing.assert_array_equal(arrays["keypoint_valid"][labeled], loss["keypoint_valid"][labeled])
    np.testing.assert_allclose(arrays["keypoints"][labeled], loss["keypoints"][labeled])
    assert arrays["observed"][2].tolist() == arrays["observed"][0].tolist()  # existence copies nearest label


def test_root_only_roots_get_a_finite_conservative_sigma():
    tracks, tracklets, alignment, cams = scene()
    tracks["tracks"][0]["positions"][1]["keypoints"] = [None] * 17   # frame 5: root located, no pose
    cfg = GoldenConfig(root_only_sigma_mm=150.0)
    arrays = build_targets(tracks, tracklets, alignment, cams, cfg)
    assert arrays["root_valid"][5, 0] and not arrays["keypoint_valid"][5, 0].any()
    assert arrays["root_sigma"][5, 0] == np.float32(150.0)
    assert np.isfinite(arrays["root_sigma"][arrays["root_valid"]]).all()


def test_golden_outliers_body_consistency_spikes_and_floor_but_never_height_alone():
    tracks, tracklets, alignment, cams = scene()
    positions = tracks["tracks"][0]["positions"]
    for pos in positions:  # the whole person hangs 60 inches up (pull-up bar / rope): kept
        pos["keypoints"] = (np.array(pos["keypoints"]) + [0, 0, 60]).tolist()
        pos["xyz"] = (np.array(pos["xyz"]) + [0, 0, 60]).tolist()
    positions[0]["keypoints"][9] = [0.0, 0.0, 200.0]      # a lone wrist far above its elbow
    positions[1]["keypoints"][15][2] += 30.0                # ankle glitch at frame 5 only (~76 cm)
    positions[2]["keypoints"][16] = [4.0, 0.0, -10.0]       # ankle underground (relative to z=0)
    arrays = build_targets(tracks, tracklets, alignment, cams, GoldenConfig())
    valid = arrays["keypoint_valid"]
    wrist, l_ankle, r_ankle = (JOINT_NAMES.index(n) for n in ("left_wrist", "left_ankle", "right_ankle"))
    assert valid[0, 0].sum() == J - 1 and not valid[0, 0, wrist]
    assert valid[5, 0].sum() == J - 1 and not valid[5, 0, l_ankle]
    assert valid[10, 0].sum() == J - 1 and not valid[10, 0, r_ankle]
    assert arrays["root_valid"][[0, 5, 10], 0].all()                         # high bodies are fine
    rejected = arrays["rejected"]   # two cameras = weak geometry: suspicious joints are dropped
    assert rejected["keypoints_body_inconsistent_dropped"] == 1 and rejected["keypoints_spikes_dropped"] == 1
    assert rejected["keypoints_below_floor"] == 1
    # Well observed (here: weak_views=1): the same suspicions only raise uncertainty.
    strong = build_targets(tracks, tracklets, alignment, cams, GoldenConfig(weak_views=1))
    assert strong["keypoint_valid"][0, 0, wrist] and strong["keypoint_valid"][5, 0, l_ankle]
    assert strong["keypoint_sigma"][0, 0, wrist] > 1000.0                    # >= its bone deviation (~1 m)
    assert strong["rejected"]["keypoints_spikes_kept_well_observed"] == 1


def test_noisy_l1_tolerates_errors_within_sigma():
    settings = LossSettings(noise_tolerance=1.0, sigma_ref_mm=30.0)
    err = torch.tensor([[0.01, 0.03, 0.10]])
    torch.testing.assert_close(noisy_l1(err, None, None, settings), err)          # exact GT: plain L1
    out = noisy_l1(err, torch.full((1, 3), 0.02), torch.tensor([30.0]), settings)
    torch.testing.assert_close(out, torch.tensor([[0.0, 0.005, 0.04]]))


def two_d_frame():
    """One person with a 3-D target, one seen in 2-D only, one present without geometry."""
    P, N = 3, 1
    gt = torch.full((1, 1, P, J, 3), 0.5)
    valid = torch.ones(1, 1, P, dtype=torch.bool)
    joints = torch.zeros(1, 1, P, J, dtype=torch.bool)
    joints[0, 0, 0] = True
    world2img = torch.eye(4)[None, None, None].repeat(1, 1, N, 1, 1)
    world2img[..., 2, 3] = 1000.0
    world2img[..., 0, 0] = world2img[..., 1, 1] = 1100.0   # focal: (100, 100, 100) mm -> pixel (100, 100)
    kp2d = torch.zeros(1, 1, N, P, J, 3)
    kp2d_valid = torch.zeros(1, 1, N, P, J, dtype=torch.bool)
    kp2d[0, 0, 0, 1, :, :2] = 100.0
    kp2d[0, 0, 0, 1, :, 2] = 1.0
    kp2d_valid[0, 0, 0, 1] = True
    targets = {"gt": gt, "valid": valid, "keypoint_valid": joints, "frame_valid": torch.ones(1, 1, dtype=torch.bool),
               "root": gt[..., :2, :].mean(-2), "root_valid": torch.zeros(1, 1, P, dtype=torch.bool),
               "kp2d": kp2d, "kp2d_valid": kp2d_valid, "world2img": world2img}
    return Supervision(targets, torch.tensor([0.0, 0, 0, 1000, 1000, 1000])), targets


def test_two_d_only_people_are_matched_by_reprojection_and_unlocated_are_not():
    sup, _ = two_d_frame()
    frame = sup.frame(0, 0)
    assert frame.located.tolist() == [True, False, False] and frame.matchable.tolist() == [True, True, False]
    preds = torch.full((4, J, 3), 0.5)
    preds[2] = 0.1   # (100, 100, 100) mm projects to pixel (100, 100): person 1's 2-D joints
    costs = {"class": 2.0, "kpt": 5.0, "alpha": 0.25, "gamma": 2.0, "root": 5.0, "root_labeled": 0.0}
    queries, people = match(frame, torch.zeros(4, 1), preds, [0, 1, 2], costs)
    assert dict(zip(people, queries))[1] == 2 and 2 not in people
    uv, _ = project(preds[2:3], frame.world2img, frame.range_min, frame.range_size)
    assert (uv - 100).abs().max() < 1e-3


def test_match_only_people_are_matched_by_interpolated_root_without_position_loss():
    _, t = two_d_frame()
    t = dict(t)
    t["frame_valid"] = torch.zeros(1, 1, dtype=torch.bool)            # a frame between labels
    t["kp2d_valid"] = torch.zeros_like(t["kp2d_valid"])
    t["keypoint_valid"] = torch.zeros_like(t["keypoint_valid"])
    match_root = torch.full((1, 1, 3, 3), 0.5)
    match_root[0, 0, 2] = 0.1                                         # person 2 is near (100, 100, 100) mm
    t["match_root"], t["match_root_valid"] = match_root, torch.tensor([[[False, False, True]]])
    sup = Supervision(t, torch.tensor([0.0, 0, 0, 1000, 1000, 1000]))
    frame = sup.frame(0, 0)
    assert frame.matchable.tolist() == [False, False, True] and not frame.located.any()
    preds = torch.full((4, J, 3), 0.5)
    preds[3] = 0.1
    costs = {"class": 2.0, "kpt": 5.0, "alpha": 0.25, "gamma": 2.0, "root": 5.0, "root_labeled": 0.0}
    queries, people = match(frame, torch.zeros(4, 1), preds, [0, 1, 2], costs)
    assert people == [2] and queries == [3]                           # assigned by the interpolated root
    out = pair_losses(frame, preds.requires_grad_(), queries, people, LossSettings())
    assert out["n_kpt"] == out["n_root"] == out["n_reproj"] == 0      # never a position target


def test_reprojection_loss_supervises_untriangulated_people():
    sup, _ = two_d_frame()
    frame = sup.frame(0, 0)
    preds = torch.full((2, J, 3), 0.3, requires_grad=True)
    out = pair_losses(frame, preds, [0, 1], [0, 1], LossSettings(reproj_tolerance_px=1.0))
    assert out["n_kpt"] == J and out["n_reproj"] == J and out["reproj"] > 0
    (out["reproj"] + out["kpt"]).backward()
    assert preds.grad[1].abs().sum() > 0


def test_criterion_never_treats_seen_people_as_absent():
    _, t = two_d_frame()
    cls = torch.zeros(1, 1, 1, 6, 1, requires_grad=True)
    kpts = torch.full((1, 1, 1, 6, J, 3), 0.5, requires_grad=True)
    loss, metrics = SetCriterion(ModelConfig())(
        cls, kpts, t["gt"], t["valid"], gt_keypoint_valid=t["keypoint_valid"], gt_root_valid=t["root_valid"],
        gt_kp2d=t["kp2d"], gt_kp2d_valid=t["kp2d_valid"], world2img=t["world2img"],
        position_range=torch.tensor([0.0, 0, 0, 1000, 1000, 1000]))
    loss.backward()
    assert metrics["num_pos"] == 3 and metrics["num_reproj"] == J
    positives = (cls.grad[0, 0, 0, :, 0] < 0).sum()       # pushed up = matched as present
    assert positives == 3                                  # 3-D, 2-D-only and unlocated person all positive


def test_camera_dropout_turns_unseen_people_into_present_but_unobserved():
    T, N, P = 2, 3, 2
    sample = {"imgs": torch.zeros(T, N, 3, 4, 4), "world2img": torch.eye(4).repeat(T, N, 1, 1),
              "intrinsics": torch.eye(4).repeat(T, N, 1, 1), "gt_valid": torch.ones(T, P, dtype=torch.bool),
              "gt_keypoint_valid": torch.ones(T, P, J, dtype=torch.bool), "gt_root_valid": torch.ones(T, P, dtype=torch.bool),
              "visibility": torch.zeros(T, N, P), "camera_ids": [0, 4, 6]}
    sample["visibility"][:, 0, 0] = 1.0   # person 0 only in view 0
    sample["visibility"][:, 2, 1] = 1.0   # person 1 only in view 2
    out = drop_cameras(sample, [2])
    assert out["imgs"].shape[1] == 1 and out["camera_ids"] == [6]
    assert out["gt_valid"][:, 0].tolist() == [False, False] and out["gt_valid"][:, 1].all()
    assert out["instance_exists"].all() and not out["gt_keypoint_valid"][:, 0].any()


def test_eval_matches_two_d_only_people_by_reprojection():
    sup, t = two_d_frame()
    keep = torch.tensor([True, True, False])
    preds = torch.full((3, J, 3), 0.5)
    preds[1] = 0.1
    keep2, pairs = _match_two_d(keep, preds, t["kp2d"][0, 0], t["kp2d_valid"][0, 0], t["world2img"][0, 0],
                                torch.tensor([1]), torch.tensor([0.0, 0, 0, 1000, 1000, 1000]), 40.0)
    assert pairs == [(1, 1)] and keep2.tolist() == [True, False, False]


def test_per_frame_camera_mask_hides_views_and_keeps_unseen_people_present():
    from unified_perception.data.augment import frame_camera_masks, mask_cameras
    from unified_perception.data.collate import collate_batch
    import random
    T, N, P = 3, 3, 2
    sample = {"imgs": torch.zeros(T, N, 3, 4, 4), "world2img": torch.eye(4).repeat(T, N, 1, 1),
              "intrinsics": torch.eye(4).repeat(T, N, 1, 1), "gt_valid": torch.ones(T, P, dtype=torch.bool),
              "gt_keypoints": torch.zeros(T, P, J, 3),
              "gt_keypoint_valid": torch.ones(T, P, J, dtype=torch.bool), "gt_root_valid": torch.ones(T, P, dtype=torch.bool),
              "visibility": torch.zeros(T, N, P), "gt_kp2d": torch.zeros(T, N, P, J, 3),
              "gt_kp2d_valid": torch.ones(T, N, P, J, dtype=torch.bool), "seq": "s", "start": 0}
    sample["visibility"][:, 0, 0] = 1.0   # person 0 only in view 0
    sample["visibility"][:, 2, 1] = 1.0   # person 1 only in view 2
    mask = torch.tensor([[True, False, False], [False, False, True], [True, True, True]])
    out = mask_cameras(sample, mask)
    assert out["imgs"].shape[1] == N                                   # shapes unchanged
    assert out["gt_valid"].tolist() == [[True, False], [False, True], [True, True]]
    assert out["instance_exists"].all()
    assert not out["gt_kp2d_valid"][0, 1:].any() and out["gt_kp2d_valid"][2].all()
    batch = collate_batch([out])
    assert batch["camera_valid"][0].tolist() == mask.tolist()
    masks = frame_camera_masks(64, 7, random.Random(0), 1)
    counts = masks.sum(1)
    assert counts.min() >= 1 and counts.max() <= 6 and len({tuple(m.tolist()) for m in masks}) > 10


def test_dropout_lane_redraws_views_every_window():
    from unified_perception.data.streaming import StreamComponent, StreamingDataset
    T, N, P = 4, 7, 1

    def load(key):
        return {"imgs": torch.zeros(T, N, 3, 2, 2), "gt_valid": torch.ones(T, P, dtype=torch.bool),
                "visibility": torch.ones(T, N, P)}
    comp = StreamComponent("toy", [[0, 1, 2]], load)
    ds = StreamingDataset([comp], camera_dropout_prob=1.0)
    masks = [ds[(0, k, k == 0, False, (123, k))]["camera_mask"] for k in range(3)]
    assert not torch.equal(masks[0], masks[1]) and not torch.equal(masks[1], masks[2])
    assert torch.equal(masks[0], ds[(0, 0, True, False, (123, 0))]["camera_mask"])   # deterministic
    off = StreamingDataset([comp], camera_dropout_prob=0.0)
    assert "camera_mask" not in off[(0, 0, True, False, (123, 0))]
