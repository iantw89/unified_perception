"""Auxiliary 2-D pose loss: targets from projected 3-D GT and 2-D-only points, frame masking."""

import torch

from unified_perception.models.pose2d_aux import Pose2DHead, aux2d_loss, heatmap_targets


def _camera(f=400.0, cx=320.0, cy=240.0):
    K = torch.tensor([[f, 0, cx, 0], [0, f, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    return K                                    # world = camera frame (identity extrinsics)


def test_heatmap_peaks_on_projected_cells():
    pts = torch.tensor([[[[100.0, 60.0], [17.0, 9.0]]]])               # (M=1, P=1, J=2, 2) px
    target, w = heatmap_targets(pts, torch.ones(1, 1, 2), torch.ones(1, 1, 2), (30, 40), 8, 1.0)
    assert target.shape == (1, 2, 30, 40) and w.tolist() == [[1.0, 1.0]]
    assert target[0, 0, 7, 12] == 1 and target[0, 1, 1, 2] == 1        # floor(px / 8)


def _batch(frame_labelled: bool):
    B, T, N, P, J = 1, 1, 1, 2, 3
    gt = torch.zeros(B, T, P, J, 3)
    gt[0, 0, 0] = torch.tensor([[0.0, 0.0, 2000.0], [200.0, 0.0, 2000.0], [0.0, 200.0, 2000.0]])
    kv = torch.zeros(B, T, P, J, dtype=torch.bool)
    kv[0, 0, 0] = True
    kp2d = torch.zeros(B, T, N, P, J, 3)
    kp2d[0, 0, 0, 1, 0, :2] = torch.tensor([500.0, 400.0])              # a 2-D-only person
    kp2d_valid = torch.zeros(B, T, N, P, J, dtype=torch.bool)
    kp2d_valid[0, 0, 0, 1, 0] = True
    return dict(gt_mm=gt, keypoint_valid=kv, frame_valid=torch.tensor([[frame_labelled]]),
                world2img=_camera().expand(B, T, N, 4, 4), camera_valid=torch.ones(B, T, N, dtype=torch.bool),
                image_hw=torch.tensor([[[480, 640]]]), image_size=(480, 640), kp2d=kp2d, kp2d_valid=kp2d_valid)


def test_loss_supervises_labelled_frames_only():
    torch.manual_seed(0)
    head = Pose2DHead(16, 3, hidden=16)
    feat = torch.randn(1, 16, 60, 80, requires_grad=True)
    loss, num_pos = aux2d_loss(head, feat, 8, 1.0, **_batch(True))
    assert num_pos.item() == 4                    # three projected joints + one 2-D-only joint
    loss.backward()
    assert torch.isfinite(loss) and feat.grad.abs().sum() > 0
    loss0, num0 = aux2d_loss(head, feat.detach(), 8, 1.0, **_batch(False))
    assert num0.item() == 0 and loss0.item() == 0  # unlabelled frame: nothing supervised


def test_person_crop_boxes_finite_for_unusable_views():
    """A view with no joint inside the image gets weight 0 and a finite placeholder box (the
    external re-ID crop budget may sample it; inf boxes made NaN crops)."""
    from unified_perception.models.reid import person_crop_boxes
    kp = torch.zeros(1, 2, 12, 3)
    kp[0, 0, :, 2] = 2000.0
    kp[0, 0, :, 0] = torch.linspace(-100, 100, 12)
    kp[0, 1, :, 2] = -2000.0                                   # behind the camera
    boxes, weight = person_crop_boxes(kp, _camera().expand(1, 1, 4, 4), (480, 640))
    assert weight.tolist() == [[[12.0], [0.0]]]
    assert torch.isfinite(boxes).all() and boxes[0, 1, 0].tolist() == [0.0, 0.0, 1.0, 1.0]
