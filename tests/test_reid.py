"""Re-identification: contrastive loss, inference gallery, retroactive metrics, head wiring."""

import numpy as np
import torch

from unified_perception.config import ModelConfig
from unified_perception.engine.metrics import IdentityAccumulator
from unified_perception.models.losses import SetCriterion
from unified_perception.models.reid import ReIDLoss, sample_appearance
from unified_perception.models.sparse_head import SparseHead
from unified_perception.models.tracking import TrackManager, TrackState

from test_tracking import RANGE, manager, outputs, person


def test_reid_loss_uses_bank_positives_across_windows_and_resets_per_lane():
    loss_fn = ReIDLoss(temperature=0.1, recent=4, anchors=2, stride=1)
    codes = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1).requires_grad_()
    loss, anchors = loss_fn([(codes, torch.tensor([1, 2]), torch.tensor([0, 0]))], torch.tensor([True]), 8)
    assert loss is None and anchors == 0          # one sighting per identity: no positives yet
    later = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1).requires_grad_()
    loss, anchors = loss_fn([(later, torch.tensor([1, 2]), torch.tensor([3, 3]))], torch.tensor([False]), 8)
    assert anchors == 2 and torch.isfinite(loss)  # positives from the previous window's bank
    loss.backward()
    assert later.grad is not None and later.grad.abs().sum() > 0
    loss, _ = loss_fn([(later.detach(), torch.tensor([1, 2]), torch.tensor([0, 0]))], torch.tensor([True]), 8)
    assert loss is None                           # a lane reset empties its bank


def test_reid_loss_keeps_first_sightings_as_long_gap_anchors():
    loss_fn = ReIDLoss(temperature=0.1, recent=2, anchors=1, stride=1)
    first = torch.tensor([[1.0, 0.0]])
    loss_fn([(first, torch.tensor([7]), torch.tensor([0]))], torch.tensor([True]), 1)
    for _ in range(10):  # many later sightings overflow the recent buffer
        loss_fn([(torch.tensor([[0.0, 1.0]]), torch.tensor([7]), torch.tensor([0]))], torch.tensor([False]), 1)
    codes, ids = loss_fn._bank(0)
    assert len(codes) == 3 and torch.equal(codes[0], first[0])


def test_appearance_sampling_averages_visible_views_only():
    tokens = torch.zeros(1, 2 * 4, 3)
    tokens[0, :4] = 1.0      # view 0 features are ones, view 1 zeros
    world2img = torch.eye(4).repeat(1, 2, 1, 1)
    world2img[0, :, 2, 3] = 1.0           # depth = z + 1
    world2img[0, 1, 0, 3] = 1000.0        # view 1 projects outside the image
    kpts = torch.tensor([4.0, 4.0, 1.0]).expand(1, 1, 2, 3)  # projects to the image centre (2, 2)
    feats = sample_appearance(tokens, [(2, 2)], kpts, world2img, (4, 4))
    torch.testing.assert_close(feats, torch.ones(1, 1, 3))


def test_inference_reid_recovers_identity_after_wait_and_rejects_strangers():
    tm = manager(reid_dims=4, reid_wait_frames=2, reid_threshold=0.8, reid_gallery_size=8,
                 track_max_inactive=0)
    state = TrackState.empty(1, 2, 8, "cpu", torch.float32, reid_dims=4, gallery=8)
    a, stranger = torch.tensor([1.0, 0, 0, 0]), torch.tensor([0, 1.0, 0, 0])
    pose = person(.2, .2)

    def step(state, scores, code):
        cls, kpts, feat = outputs(3, 2, {0: pose, 3: pose, 4: pose}, scores)
        codes = torch.zeros(1, 5, 4)
        codes[0, 0] = code
        return tm.infer_step(state, 3, cls, kpts, feat, codes=codes, ema=torch.stack([code, code])[None])

    reported, state, _ = step(state, {0: .9}, a)             # id 0 born (provisional)
    for _ in range(2):
        reported, state, _ = step(state, {3: .9}, a)         # lives past the wait
    reported, state, _ = step(state, {}, a)                  # disappears; expires into the gallery
    assert not state.valid.any() and 0 in state.gallery_ids[0][state.gallery_valid[0]].tolist()
    reported, state, aliases = step(state, {0: .9}, a)       # reborn as provisional id 1
    assert reported[0, 0] == 1 and aliases == []
    reported, state, aliases = step(state, {3: .9}, a)       # wait reached: takes over id 0
    assert aliases == [(0, 1, 0)] and reported[0, 3] == 0
    reported, state, _ = step(state, {}, a)
    reported, state, _ = step(state, {0: .9}, stranger)      # someone else
    reported, state, aliases = step(state, {3: .9}, stranger)
    assert aliases == [] and reported[0, 3] == 2


def test_identity_metrics_resolve_aliases_retroactively():
    acc = IdentityAccumulator()
    acc.reset_clip(0)
    for label in (5, 5, 9, 9):  # same person, provisional id 9 after a gap
        acc.update(np.array([label]), np.array([42]), [(0, 0)], lane=0)
    acc.alias(0, 9, 5)
    metrics = acc.compute()
    assert metrics["id_switches"] == 0 and metrics["id_switches_online"] == 1
    assert metrics["id_reid_takeovers"] == 1


def test_head_with_reid_trains_identity_codes():
    cfg = ModelConfig(track_queries=True, max_tracks=3, track_drop_prob=0.0, track_fp_prob=0.0,
                      reid_dims=8, reid_bank_stride=1)
    head = SparseHead(embed_dims=24, num_joints=12, num_query=8, num_propagated=2,
                      memory_len=8, topk_proposals=4, num_layers=2, num_heads=2,
                      ffn_dim=32, dropout=0., num_cams=2, num_levels=1, num_pts=1,
                      position_range=RANGE, scalar=2, max_tracks=3,
                      track_manager=TrackManager(cfg, RANGE), reid_dims=8)
    B, T = 1, 4
    feats = torch.randn(B, T, 8, 24, requires_grad=True)
    codes = torch.randn(B, T, 8, 14)
    projection = torch.eye(4).repeat(B, T, 2, 1, 1)
    projection[..., 2, 3] = 30.0  # positive depth so keypoints project into the views
    gt = torch.rand(B, T, 4, 12, 3) * .5 + .25
    valid = torch.zeros(B, T, 4, dtype=torch.bool)
    valid[:, :, :2] = True
    targets = {"gt": gt, "valid": valid, "keypoint_valid": valid[..., None].expand(B, T, 4, 12),
               "frame_valid": torch.ones(B, T, dtype=torch.bool), "root": gt[..., 6:8, :].mean(-2),
               "root_valid": valid, "instance_ids": torch.tensor([3, 5, -1, -1]).expand(B, T, 4),
               "exists": valid}
    cls, kpts, dn = head(feats, codes, projection, [(2, 2)], (16, 16), gt, valid, track_targets=targets)
    out = head.track_output
    assert out["id_codes"].shape == (B, T, 11, 8) and out["id_ema"].shape == (B, T, 3, 8)
    loss, metrics = SetCriterion(cfg)(cls, kpts, gt, valid, dn, track_output=out)
    assert metrics["num_reid_anchors"] > 0 and "loss_reid" in metrics
    loss.backward()
    assert head.identity_head.query_proj[0].weight.grad.abs().sum() > 0
    assert head.identity_head.app_proj[0].weight.grad.abs().sum() > 0
    head.eval()
    with torch.no_grad():
        head(feats, codes, projection, [(2, 2)], (16, 16))
    assert head.track_output["aliases"] == [] and head.track_state.gallery_code.shape == (B, 128, 8)
