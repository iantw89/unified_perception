import numpy as np
import pytest
import torch
from torch import nn

from unified_perception.config import Config
from unified_perception.data.collate import collate_batch
from unified_perception.engine.evaluate import run_eval
from unified_perception.engine.step import TrainStep
from unified_perception.models import detector
from unified_perception.models.losses import SetCriterion
from unified_perception.models.transformer import Detr3DCrossAttn
from test_msda_parity import _make_inputs, _mmcv_available


class TinyBackbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, stride=8, padding=1)
        self.batch_sizes = []

    def forward(self, images, return_detections=False):
        self.batch_sizes.append(len(images))
        detections = [
            {"boxes": np.ones((1, 4), np.float32),
             "kpts": np.ones((1, 12, 2), np.float32),
             "scores": np.ones(1, np.float32)}
            for _ in images
        ] if return_detections else None
        return [self.conv(images)], detections


def sample(cameras, frames=2):
    calibration = torch.eye(4).repeat(frames, cameras, 1, 1)
    calibration[..., 0, 0] = calibration[..., 1, 1] = 16
    calibration[..., 0, 2] = calibration[..., 1, 2] = 16
    keypoints = torch.zeros(frames, 2, 12, 3)
    keypoints[..., 2] = 1000
    return dict(
        imgs=torch.randint(0, 256, (frames, cameras, 3, 32, 32), dtype=torch.uint8),
        world2img=calibration, intrinsics=calibration.clone(),
        gt_keypoints=keypoints, gt_valid=torch.ones(frames, 2, dtype=torch.bool),
        visibility=torch.ones(frames, cameras, 2),
        instance_ids=torch.arange(2).expand(frames, -1), seq="test", start=0,
    )


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(detector, "YOLOPoseBackbone", TinyBackbone)
    torch.manual_seed(7)
    cfg = Config()
    cfg.data.num_cameras = 7
    cfg.data.max_people = 2
    cfg.model.position_range = [-10, -10, 500, 10, 10, 1500]
    cfg.model.embed_dims = 24
    cfg.model.num_queries = 4
    cfg.model.dec_layers = 2
    cfg.model.dec_heads = 2
    cfg.model.dec_groups = 2
    cfg.model.dec_ffn_dim = 32
    cfg.model.dec_num_pts = 2
    cfg.model.num_propagated = 2
    cfg.model.memory_len = 4
    cfg.model.memory_topk = 2
    cfg.model.dropout = 0
    cfg.model.dn_scalar = 2
    cfg.model.use_mmcv_msda = False
    result = detector.UnifiedPerceptionModel(cfg).eval()
    for layer in result.head.decoder.layers:
        nn.init.normal_(layer.cross_attn.weights_fc.weight, std=0.1)
    return result


def test_collate_pads_only_cameras_and_preserves_supervision():
    samples = [sample(n) for n in (4, 6, 7, 4)]
    batch = collate_batch(samples)
    assert batch["imgs"].shape == (4, 2, 7, 3, 32, 32)
    assert batch["camera_valid"].dtype == torch.bool
    for b, source in enumerate(samples):
        n = source["imgs"].shape[1]
        assert batch["camera_valid"][b, :, :n].all()
        assert not batch["camera_valid"][b, :, n:].any()
        assert not batch["imgs"][b, :, n:].any()
        assert not batch["visibility"][b, :, n:].any()
        for key in ("imgs", "intrinsics", "world2img", "visibility"):
            torch.testing.assert_close(batch[key][b, :, :n], source[key])
        for key in ("gt_keypoints", "gt_valid", "instance_ids"):
            torch.testing.assert_close(batch[key][b], source[key])
    assert torch.isfinite(torch.linalg.inv(batch["intrinsics"])).all()
    uniform = collate_batch([sample(4), sample(4)])
    assert uniform["camera_valid"].all()
    assert uniform["imgs"].shape[2] == 4


def test_masked_batch_matches_unpadded_temporal_predictions(model):
    samples = [sample(n) for n in (4, 6, 7, 4)]
    batch = collate_batch(samples)
    with torch.no_grad():
        scores, poses, _, detections = model(
            batch["imgs"].float(), batch["world2img"], batch["intrinsics"],
            camera_valid=batch["camera_valid"], return_2d=True,
        )
        assert model.backbone.batch_sizes[-2:] == [21, 21]
        for b, source in enumerate(samples):
            separate = model(source["imgs"][None].float(), source["world2img"][None],
                             source["intrinsics"][None])
            torch.testing.assert_close(scores[:, b:b+1], separate[0], atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(poses[:, b:b+1], separate[1], atol=2e-6, rtol=2e-5)
            for t in range(2):
                for n in range(7):
                    det = detections[t * 4 * 7 + b * 7 + n]
                    assert len(det["boxes"]) == int(n < source["imgs"].shape[1])
        imgs = batch["imgs"].float().masked_fill(~batch["camera_valid"][..., None, None, None], float("nan"))
        calibration_mask = ~batch["camera_valid"][..., None, None]
        changed = model(
            imgs, batch["world2img"].masked_fill(calibration_mask, float("nan")),
            batch["intrinsics"].masked_fill(calibration_mask, 0),
            camera_valid=batch["camera_valid"],
        )
        torch.testing.assert_close(scores, changed[0])
        torch.testing.assert_close(poses, changed[1])


def test_masked_training_step_has_finite_gradients_and_zero_padding_gradients(model):
    model.train()
    batch = collate_batch([sample(n) for n in (4, 6, 7, 4)])
    batch["imgs"] = batch["imgs"].float().requires_grad_()
    # TrainStep normalizes in-place; make the input non-leaf while retaining its gradient.
    images = batch["imgs"]
    batch["imgs"] = images + 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    step = TrainStep(model, SetCriterion(model.cfg.model), optimizer,
                     torch.zeros(1, 1, 1, 3, 1, 1), torch.ones(1, 1, 1, 3, 1, 1),
                     torch.device("cpu"), False)
    step.h2d(batch)
    step.normalize()
    step.forward()
    assert np.isfinite(step.compute_loss())
    step.free_inputs()
    assert np.isfinite(step.backward())
    assert torch.isfinite(images.grad).all()
    assert not images.grad[~batch["camera_valid"]].any()
    assert images.grad[batch["camera_valid"]].abs().sum() > 0
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    step.optimize(1)


def test_eval_uses_real_camera_count_buckets(model):
    cfg = model.cfg
    batch = collate_batch([sample(4), sample(7)])
    metrics = run_eval(model, [batch], cfg, torch.device("cpu"))
    assert any(key.startswith("cameras_4/") for key in metrics)
    assert any(key.startswith("cameras_7/") for key in metrics)


def test_all_out_of_view_attention_has_finite_backward():
    attention = Detr3DCrossAttn(8, num_cams=7, num_levels=1, num_pts=1, dropout=0)
    query = torch.randn(1, 2, 8, requires_grad=True)
    feats = torch.randn(7, 4, 8, requires_grad=True)
    output = attention(query, torch.zeros_like(query), feats, torch.full((1, 2, 3), 0.5),
                       torch.tensor([-1., -1., -20., 1., 1., -10.]),
                       torch.eye(4).repeat(1, 7, 1, 1), [(2, 2)], (32, 32),
                       camera_valid=torch.tensor([[True, True, True, True, False, False, False]]))
    output.sum().backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(query.grad).all()
    assert torch.isfinite(feats.grad).all()
    assert not feats.grad.any()


def test_model_rejects_missing_views_or_invalid_mask_shape(model):
    batch = collate_batch([sample(4)])
    for mask, error in ((torch.zeros(1, 2, 4, dtype=torch.bool), "at least one"),
                        (torch.ones(1, 4, dtype=torch.bool), "shape"),
                        (torch.ones(1, 2, 4), "boolean")):
        with pytest.raises(ValueError, match=error):
            model(batch["imgs"].float(), batch["world2img"], batch["intrinsics"], camera_valid=mask)


@pytest.mark.parametrize("use_mmcv", [False, True])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_attention_mask_matches_unpadded_cameras(use_mmcv):
    if use_mmcv and not _mmcv_available():
        pytest.skip("MMCV CUDA kernel unavailable")
    attention = Detr3DCrossAttn(16, 7, 2, num_pts=2, num_groups=2,
                               dropout=0, use_mmcv_msda=use_mmcv).cuda()
    nn.init.normal_(attention.weights_fc.weight, std=0.1)
    inputs = _make_inputs(B=2, N=7, Q=4, C=16, spatial_shapes=[(4, 4), (2, 2)])
    inputs["feats"].requires_grad_()
    mask = torch.tensor([[True] * 4 + [False] * 3, [True] * 7], device="cuda")
    output = attention(**inputs, camera_valid=mask)
    for b, cameras in enumerate((4, 7)):
        separate = dict(inputs)
        for key in ("query", "query_pos", "reference_points"):
            separate[key] = inputs[key][b:b+1]
        separate["world2img"] = inputs["world2img"][b:b+1, :cameras]
        separate["feats"] = inputs["feats"].view(2, 7, 20, 16)[b, :cameras]
        torch.testing.assert_close(output[b:b+1], attention(**separate), atol=1e-5, rtol=1e-4)
    output.square().sum().backward()
    gradients = inputs["feats"].grad.view(2, 7, 20, 16)
    assert torch.isfinite(gradients).all()
    assert not gradients[~mask].any()
    assert gradients[mask].abs().sum() > 0
