"""Pretrained crop re-ID branch with a tiny stand-in TorchScript network."""

import torch

from unified_perception.config import ModelConfig
from unified_perception.models.losses import SetCriterion
from unified_perception.models.reid import ExternalReID, person_crop_boxes
from unified_perception.models.sparse_head import SparseHead
from unified_perception.models.tracking import TrackManager

from test_tracking import RANGE


class MeanColour(torch.nn.Module):
    """Embedding = normalized mean RGB of the crop (a 3-d 'appearance')."""

    def forward(self, images):
        return torch.nn.functional.normalize(images.mean(dim=(2, 3)) + 1e-3, dim=-1)


def scripted(tmp_path):
    path = tmp_path / "reid.ts"
    torch.jit.script(MeanColour()).save(str(path))
    return str(path)


def test_crop_boxes_follow_projection_aspect_and_visibility():
    world2img = torch.eye(4).repeat(1, 2, 1, 1)
    world2img[0, :, 2, 3] = 1.0
    world2img[0, 1, 0, 3] = 5000.0  # second view sees nothing
    kpts = torch.zeros(1, 1, 4, 3)
    kpts[0, 0, :, 0] = torch.tensor([10.0, 12.0, 10.0, 12.0])
    kpts[0, 0, :, 1] = torch.tensor([10.0, 10.0, 30.0, 30.0])
    boxes, weight = person_crop_boxes(kpts, world2img, (100, 100))
    assert weight.tolist() == [[[4.0, 0.0]]]
    x0, y0, x1, y1 = boxes[0, 0, 0].tolist()
    assert abs((y1 - y0) / (x1 - x0) - 3.0) < 1e-4 and y0 < 10 and y1 > 30


def test_external_reid_separates_differently_dressed_people(tmp_path):
    reid = ExternalReID(scripted(tmp_path), crop_hw=(12, 4))
    images = torch.zeros(1, 1, 3, 64, 64)
    images[0, 0, 0, :, :32] = 255.0   # red person on the left
    images[0, 0, 2, :, 32:] = 255.0   # blue person on the right
    world2img = torch.eye(4)[None, None].clone()
    world2img[0, 0, 2, 3] = 1.0
    kpts = torch.zeros(1, 2, 4, 3)
    for person, x in enumerate((14.0, 48.0)):
        kpts[0, person, :, 0] = torch.tensor([x - 2, x + 2, x - 2, x + 2])
        kpts[0, person, :, 1] = torch.tensor([20.0, 20.0, 44.0, 44.0])
    codes, ok = reid(images, kpts, world2img)
    assert ok.all()
    assert codes[0, 0, 0] > 0.9 and codes[0, 1, 2] > 0.9 and float(codes[0, 0] @ codes[0, 1]) < 0.2


def test_head_fuses_crop_embeddings_into_track_identity(tmp_path):
    cfg = ModelConfig(track_queries=True, max_tracks=3, track_drop_prob=0.0, track_fp_prob=0.0,
                      reid_dims=8, reid_bank_stride=1)
    head = SparseHead(embed_dims=24, num_joints=12, num_query=8, num_propagated=2,
                      memory_len=8, topk_proposals=4, num_layers=2, num_heads=2,
                      ffn_dim=32, dropout=0., num_cams=2, num_levels=1, num_pts=1,
                      position_range=RANGE, scalar=2, max_tracks=3,
                      track_manager=TrackManager(cfg, RANGE), reid_dims=8,
                      external_reid=ExternalReID(scripted(tmp_path), crop_hw=(12, 4)), external_stride=1)
    assert not any("external" in k for k in head.state_dict())
    B, T = 1, 4
    feats = torch.randn(B, T, 8, 24, requires_grad=True)
    codes = torch.randn(B, T, 8, 14)
    projection = torch.eye(4).repeat(B, T, 2, 1, 1)
    projection[..., 2, 3] = 30.0
    images = torch.rand(B, T, 2, 3, 16, 16)
    gt = torch.rand(B, T, 4, 12, 3) * .5 + .25
    valid = torch.zeros(B, T, 4, dtype=torch.bool)
    valid[:, :, :2] = True
    targets = {"gt": gt, "valid": valid, "keypoint_valid": valid[..., None].expand(B, T, 4, 12),
               "frame_valid": torch.ones(B, T, dtype=torch.bool), "root": gt[..., 6:8, :].mean(-2),
               "root_valid": valid, "instance_ids": torch.tensor([3, 5, -1, -1]).expand(B, T, 4),
               "exists": valid}
    cls, kpts, dn = head(feats, codes, projection, [(2, 2)], (16, 16), gt, valid,
                         images=images, track_targets=targets)
    assert head.track_state.app_ema.shape == (B, 3, 3)
    loss, metrics = SetCriterion(cfg)(cls, kpts, gt, valid, dn, track_output=head.track_output)
    loss.backward()
    assert head.identity_head.ext_proj[0].weight.grad is not None
