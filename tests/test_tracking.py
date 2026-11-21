"""Track queries (models/tracking.py), their loss and streaming batches."""

import pytest
import torch

from unified_perception.config import ModelConfig
from unified_perception.data.identity import fill_identity_gaps
from unified_perception.data.streaming import (
    ChunkPolicy, StreamComponent, StreamingBatchSampler, chained_windows, evaluation_lanes,
)
from unified_perception.models.losses import SetCriterion
from unified_perception.models.sparse_head import SparseHead
from unified_perception.models.tracking import (
    FALSE_TRACK_ID, TrackManager, TrackState, track_key_mask,
)

RANGE = [-10, -10, 1, 10, 10, 20]


def manager(**overrides):
    cfg = ModelConfig(track_queries=True, track_drop_prob=0.0, track_fp_prob=0.0, track_ref_noise=0.0,
                      track_birth_score=0.5, track_keep_score=0.4, track_max_inactive=2,
                      track_nms_mm=500.0)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return TrackManager(cfg, RANGE)


def person(x, y, z=0.5, joints=12):
    kpts = torch.full((joints, 3), z)
    kpts[:, 0], kpts[:, 1] = x, y
    return kpts


def frame_targets(people, ids, exists=None, P=4, T=1):
    """One lane, one frame: people are (12, 3) normalized poses."""
    gt = torch.zeros(1, T, P, 12, 3)
    valid = torch.zeros(1, T, P, dtype=torch.bool)
    instance = torch.full((1, T, P), -1, dtype=torch.long)
    for slot, (kpts, identity) in enumerate(zip(people, ids)):
        if kpts is not None:
            gt[0, :, slot] = kpts
            valid[0, :, slot] = True
        instance[0, :, slot] = identity
    root = gt[..., 6:8, :].mean(-2)
    present = valid.clone() if exists is None else exists
    return {"gt": gt, "valid": valid, "keypoint_valid": valid[..., None].expand(1, T, P, 12),
            "frame_valid": torch.ones(1, T, dtype=torch.bool), "root": root, "root_valid": valid,
            "instance_ids": instance, "exists": present}


def outputs(num_det, slots, poses, scores):
    """Final-layer outputs: per-query poses (Qm, 12, 3) and probabilities."""
    Qm = num_det + slots
    kpts = torch.full((1, Qm, 12, 3), 0.9)
    for q, pose in poses.items():
        kpts[0, q] = pose
    probs = torch.full((1, Qm, 1), 0.01)
    for q, p in scores.items():
        probs[0, q, 0] = p
    return torch.logit(probs), kpts, torch.randn(1, Qm, 8)


def test_train_step_assigns_tracks_by_identity_ignores_unobserved_and_births():
    tm = manager()
    state = TrackState.empty(1, 3, 8, "cpu", torch.float32)
    # Slot 0 tracks id 7 (present), slot 1 tracks id 8 (exists, unobserved), slot 2 id 9 (gone).
    for k, identity in enumerate([7, 8, 9]):
        state.valid[0, k], state.ids[0, k] = True, identity
    a, b, c = person(.2, .2), person(.5, .5), person(.8, .8)
    exists = torch.tensor([[[True, True, True, False]]])
    targets = frame_targets([a, None, c, None], [7, 8, 11, -1], exists=exists)
    cls, kpts, feat = outputs(4, 3, {0: c, 4: a}, {0: .9, 4: .9})
    (assign,), new = tm.train_step(state, 4, cls, kpts, feat, targets, 0)
    assert assign["track_q"] == [4] and assign["track_g"] == [0]
    assert assign["ignore"] == [5] and assign["background"] == [6] and assign["invalid"] == []
    assert assign["det_g"] == [2] and assign["det_q"] == [0]       # only the unclaimed person
    assert new.ids[0].tolist() == [7, 8, 11] and new.valid[0].all()  # carried, carried, born
    torch.testing.assert_close(new.embed[0, 2], feat[0, 0])
    # The unobserved slot is frozen (as at inference) and ages; the observed one refreshes.
    torch.testing.assert_close(new.embed[0, 1], state.embed[0, 1])
    torch.testing.assert_close(new.ref[0, 1], state.ref[0, 1])
    assert new.inactive[0].tolist() == [0, 1, 0]
    torch.testing.assert_close(new.embed[0, 0], feat[0, 4])


def test_train_step_drop_and_false_track_augmentation():
    tm = manager(track_drop_prob=1.0, track_fp_prob=1.0)
    state = TrackState.empty(1, 2, 8, "cpu", torch.float32)
    state.valid[0, 0], state.ids[0, 0] = True, 7
    targets = frame_targets([person(.2, .2)], [7])
    cls, kpts, feat = outputs(4, 2, {4: person(.2, .2)}, {4: .9, 2: .6})
    _, new = tm.train_step(state, 4, cls, kpts, feat, targets, 0)
    assert new.ids[0].tolist() == [FALSE_TRACK_ID, -1]   # real track dropped, false track injected
    torch.testing.assert_close(new.embed[0, 0], feat[0, 2])  # highest-scoring unmatched detection


def test_infer_step_birth_keep_inactive_revive_and_expire():
    tm = manager()
    state = TrackState.empty(1, 2, 8, "cpu", torch.float32)
    a = person(.2, .2)
    cls, kpts, feat = outputs(3, 2, {0: a, 1: person(.21, .2)}, {0: .9, 1: .8})
    reported, state, _ = tm.infer_step(state, 3, cls, kpts, feat)
    assert reported[0, 0] == 0 and reported[0, 1] == -1          # duplicate within nms suppressed
    assert state.ids[0].tolist() == [0, -1]
    cls, kpts, feat = outputs(3, 2, {3: a}, {3: .45})            # track slot 0 sits at query 3
    reported, state, _ = tm.infer_step(state, 3, cls, kpts, feat)
    assert reported[0, 3] == 0                                   # keep threshold, still active
    cls, kpts, feat = outputs(3, 2, {3: a}, {3: .2})
    reported, state, _ = tm.infer_step(state, 3, cls, kpts, feat)
    assert (reported == -1).all() and state.inactive[0, 0] == 1
    cls, kpts, feat = outputs(3, 2, {3: a}, {3: .45})            # inactive needs the birth score
    reported, state, _ = tm.infer_step(state, 3, cls, kpts, feat)
    assert (reported == -1).all() and state.inactive[0, 0] == 2
    cls, kpts, feat = outputs(3, 2, {3: a}, {3: .6})
    reported, state, _ = tm.infer_step(state, 3, cls, kpts, feat)
    assert reported[0, 3] == 0 and state.inactive[0, 0] == 0     # re-identified, same id
    for _ in range(3):
        cls, kpts, feat = outputs(3, 2, {}, {})
        _, state, _ = tm.infer_step(state, 3, cls, kpts, feat)
    assert not state.valid.any()                                 # expired after max_inactive
    cls, kpts, feat = outputs(3, 2, {0: a}, {0: .9})
    reported, state, _ = tm.infer_step(state, 3, cls, kpts, feat)
    assert reported[0, 0] == 1                                   # ids never reused within a lane


def test_track_key_mask_hides_invalid_slots_and_dn_from_tracks():
    base = torch.zeros(5, 5 + 3, dtype=torch.bool)
    base[2:, :2] = True  # 2 DN queries, 3 detection queries
    valid = torch.tensor([[True, False], [False, False]])
    mask = track_key_mask(base, 5, valid, 3, num_heads=2)
    assert mask.shape == (4, 7, 10)
    lane0, lane1 = mask[0], mask[2]
    assert not lane0[:, 5].any() and lane0[:, 6].all() and lane1[:, 5:7].all()
    assert lane0[5:, :2].all() and not lane0[5:, 2:5].any()   # tracks never see DN queries
    assert not lane0[:, 7:].any()                             # memory stays visible


def test_tracked_loss_uses_assignments_and_ignores_unsupervised_slots():
    criterion = SetCriterion(ModelConfig())
    L, Qm, P = 2, 6, 3
    gt = torch.rand(1, 1, P, 12, 3)
    valid = torch.tensor([[[True, True, False]]])
    kpts = torch.rand(L, 1, 1, Qm, 12, 3, requires_grad=True)
    cls = torch.zeros(L, 1, 1, Qm, 1, requires_grad=True)
    assign = {"track_q": [4], "track_g": [1], "det_q": [0], "det_g": [0],
              "background": [], "ignore": [5], "invalid": []}
    out = {"num_det": 4, "assignments": [[assign]]}
    loss, metrics = criterion(cls, kpts, gt, valid, track_output=out)
    loss.backward()
    assert metrics["num_track_pairs"] == 1 and metrics["num_pos"] == 2
    assert cls.grad[:, 0, 0, 5].abs().sum() == 0                 # ignored slot: no class gradient
    assert kpts.grad[-1, 0, 0, 4].abs().sum() > 0                # track slot regresses its identity
    assert kpts.grad[-1, 0, 0, 0].abs().sum() > 0


def test_head_streams_track_state_across_calls_and_resets_lanes():
    cfg = ModelConfig(track_queries=True, max_tracks=3, track_drop_prob=0.0, track_fp_prob=0.0)
    head = SparseHead(embed_dims=24, num_joints=12, num_query=8, num_propagated=2,
                      memory_len=8, topk_proposals=4, num_layers=2, num_heads=2,
                      ffn_dim=32, dropout=0., num_cams=2, num_levels=1, num_pts=1,
                      position_range=RANGE, scalar=2, max_tracks=3,
                      track_manager=TrackManager(cfg, RANGE))
    B, T = 2, 3
    feats = torch.randn(B, T, 8, 24, requires_grad=True)
    codes = torch.randn(B, T, 8, 14)
    projection = torch.eye(4).repeat(B, T, 2, 1, 1)
    gt = torch.rand(B, T, 4, 12, 3) * .5 + .25
    valid = torch.zeros(B, T, 4, dtype=torch.bool)
    valid[:, :, :2] = True
    targets = {"gt": gt, "valid": valid, "keypoint_valid": valid[..., None].expand(B, T, 4, 12),
               "frame_valid": torch.ones(B, T, dtype=torch.bool), "root": gt[..., 6:8, :].mean(-2),
               "root_valid": valid, "instance_ids": torch.tensor([3, 5, -1, -1]).expand(B, T, 4),
               "exists": valid}
    cls, kpts, dn = head(feats, codes, projection, [(2, 2)], (16, 16), gt, valid, track_targets=targets)
    assert cls.shape == (2, B, T, 8 + 3, 1)
    out = head.track_output
    assert not out["slot_valid"][:, 0].any()                         # first frame: no tracks yet
    assert out["slot_valid"][:, 1].sum(-1).tolist() == [2, 2]        # both people tracked by frame 1
    assert sorted(head.track_state.ids[0][head.track_state.valid[0]].tolist()) == [3, 5]
    loss, metrics = SetCriterion(cfg)(cls, kpts, gt, valid, dn, track_output=out)
    loss.backward()
    assert torch.isfinite(feats.grad).all() and metrics["num_track_pairs"] == 2 * B * (T - 1)
    # Streaming: lane 0 continues (tracks alive at its first frame), lane 1 resets.
    head(feats.detach(), codes, projection, [(2, 2)], (16, 16), gt, valid, track_targets=targets,
         stream_reset=torch.tensor([False, True]))
    first = head.track_output["slot_valid"][:, 0].sum(-1).tolist()
    assert first == [2, 0]
    # Inference runs without targets and reports running ids.
    head.eval()
    with torch.no_grad():
        head(feats, codes, projection, [(2, 2)], (16, 16))
    assert head.track_output["track_ids"].shape == (B, T, 11) and head.track_output["assignments"] is None


def test_chained_windows_and_training_lanes():
    samples = [("a", s) for s in (0, 2, 4, 8, 12, 16)] + [("b", 0)]
    groups = chained_windows(samples, 4)
    assert [("a", 0), ("a", 4), ("a", 8), ("a", 12), ("a", 16)] in groups
    assert [("a", 2)] in groups and [("b", 0)] in groups
    component = StreamComponent("x", [list(range(i * 10, i * 10 + 5)) for i in range(3)], lambda k: {}, 1.0)
    sampler = StreamingBatchSampler([component], batch_size=2, num_batches=12,
                                    policies=[ChunkPolicy([3], [1.0])], seed=1)
    lanes = list(zip(*list(sampler)))
    for lane in lanes:
        run = []
        for _, key, reset, pad, *_ in lane:
            assert not pad
            if reset:
                run = [key]
            else:
                assert key == run[-1] + 1          # consecutive windows of one clip
                run.append(key)
            assert len(run) <= 3


def test_evaluation_lanes_cover_every_window_once_in_order():
    components = [StreamComponent("x", [[1, 2, 3], [4], [5, 6]], lambda k: {})]
    batches = evaluation_lanes(components, batch_size=2)
    real = [(key, reset) for batch in batches for _, key, reset, pad in batch if not pad]
    assert sorted(k for k, _ in real) == [1, 2, 3, 4, 5, 6]
    assert {k for k, r in real if r} == {1, 4, 5}
    assert all(len(batch) == 2 for batch in batches)
    with pytest.raises(ValueError):
        ChunkPolicy([4], [1.0, 2.0])


def test_fill_identity_gaps_only_between_observations():
    observed = torch.zeros(12, 2, dtype=torch.bool).numpy()
    observed[[1, 4, 11], 0] = True          # gaps of 2 and 6 frames
    observed[[3], 1] = True                 # single sighting: nothing to fill
    present = fill_identity_gaps(observed, max_gap=3)
    assert present[:, 0].nonzero()[0].tolist() == [1, 2, 3, 4, 11]
    assert present[:, 1].nonzero()[0].tolist() == [3]
    assert (fill_identity_gaps(observed, 0) == observed).all()


def test_partition_covers_every_window_once_with_fresh_random_cuts():
    import random
    groups = [list(range(i * 1000, i * 1000 + n)) for i, n in enumerate([700, 300, 5, 1])]
    policy = ChunkPolicy([1, 16, 0], [0.2, 0.3, 0.5])
    first = policy.partition(groups, random.Random(0))
    second = policy.partition(groups, random.Random(1))
    for pieces in (first, second):
        windows = [w for piece in pieces for w in piece]
        assert sorted(windows) == sorted(w for g in groups for w in g)    # nothing skipped or repeated
        for piece in pieces:
            assert piece == list(range(piece[0], piece[0] + len(piece)))  # consecutive windows

    assert {p[0] for p in first} != {p[0] for p in second}                # cut points change per pass
    import random as _r
    single = [list(range(500))]
    runs = [max(map(len, policy.partition(single, _r.Random(i)))) for i in range(40)]
    assert 500 in runs and min(runs) < 500                                # one long clip: sometimes whole, else cut
    assert ChunkPolicy([1], [1.0]).streaming is False and policy.streaming


def test_sampler_lanes_persist_across_epochs_and_follow_window_fractions():
    long_clip = StreamComponent("gym", [list(range(200))], lambda k: {}, 0.5)
    short = StreamComponent("mocap", [[10_000 + i] for i in range(50)], lambda k: {}, 0.5)
    # 6050 (not a multiple of the 200-window piece): a lane is mid-piece at the epoch boundary.
    sampler = StreamingBatchSampler([long_clip, short], batch_size=4, num_batches=6050,
                                    policies=[ChunkPolicy([0], [1.0]), ChunkPolicy([1], [1.0])], seed=3)
    epochs = [list(sampler), list(sampler)]
    continued = 0
    for lane in range(4):  # an hours-long piece continues into the next epoch without a reset
        c0, k0 = epochs[0][-1][lane][:2]
        c1, k1, reset = epochs[1][0][lane][:3]
        if c0 == 0 and c1 == 0 and not reset:
            assert k1 == k0 + 1
            continued += 1
    assert continued >= 1
    from_long = sum(item[0] == 0 for epoch in epochs for batch in epoch for item in batch)
    assert 0.4 < from_long / (2 * 6050 * 4) < 0.6


def test_sampler_gives_every_batch_its_lane_quota():
    # One long domain clip + short auxiliary clips (the golden-set shape that once left
    # two thirds of the steps without a single gym lane).
    gym = StreamComponent("gym", [list(range(593))], lambda k: {}, 0.5)
    mocap = StreamComponent("mocap", [[10_000 + i] for i in range(300)], lambda k: {}, 0.35)
    aligned = StreamComponent("aligned", [list(range(20_000 + 18 * i, 20_018 + 18 * i)) for i in range(40)],
                              lambda k: {}, 0.15)
    policies = [ChunkPolicy([16, 128, 1024, 0], [.1, .2, .3, .4]), ChunkPolicy([1, 4, 16, 0], [.25] * 4),
                ChunkPolicy([1, 4, 16, 0], [.25] * 4)]
    for seed in range(3):
        batches = list(StreamingBatchSampler([gym, mocap, aligned], 4, 4000, policies, seed=seed))
        per_batch = [sum(item[0] == 0 for item in batch) for batch in batches]
        assert set(per_batch) == {2}                          # gym: exactly 2 of 4 lanes, always
        shares = [sum(item[0] == c for batch in batches for item in batch) / (4 * 4000) for c in (1, 2)]
        assert abs(shares[0] - 0.35) < 0.03 and abs(shares[1] - 0.15) < 0.03


def test_shared_aux_matching_equals_per_layer_loss_when_layers_agree():
    from types import SimpleNamespace
    cfg = ModelConfig(track_queries=True, max_tracks=3, track_drop_prob=0.0, track_fp_prob=0.0)
    head = SparseHead(embed_dims=24, num_joints=12, num_query=8, num_propagated=2,
                      memory_len=8, topk_proposals=4, num_layers=2, num_heads=2,
                      ffn_dim=32, dropout=0., num_cams=2, num_levels=1, num_pts=1,
                      position_range=RANGE, scalar=2, max_tracks=3, track_manager=TrackManager(cfg, RANGE))
    B, T = 2, 3
    gt = torch.rand(B, T, 4, 12, 3) * .5 + .25
    valid = torch.zeros(B, T, 4, dtype=torch.bool)
    valid[:, :, :2] = True
    targets = {"gt": gt, "valid": valid, "keypoint_valid": valid[..., None].expand(B, T, 4, 12),
               "frame_valid": torch.ones(B, T, dtype=torch.bool), "root": gt[..., 6:8, :].mean(-2),
               "root_valid": valid, "instance_ids": torch.tensor([3, 5, -1, -1]).expand(B, T, 4), "exists": valid}
    cls, kpts, _ = head(torch.randn(B, T, 8, 24), torch.randn(B, T, 8, 14), torch.eye(4).repeat(B, T, 2, 1, 1),
                        [(2, 2)], (16, 16), gt, valid, track_targets=targets)
    same_cls, same_kpts = cls[-1:].expand_as(cls).detach(), kpts[-1:].expand_as(kpts).detach()
    losses = []
    for share in (False, True):
        criterion = SetCriterion(cfg, SimpleNamespace(root_weight=5.0, gt_noise_tolerance=1.0, gt_sigma_ref_mm=30.0,
                                                      reproj_weight=1.0, reproj_tolerance_px=4.0,
                                                      reproj_scale_px=64.0, reproj_triangulated_weight=0.0,
                                                      share_aux_matching=share))
        losses.append(criterion(same_cls, same_kpts, gt, valid, track_output=head.track_output))
    torch.testing.assert_close(losses[0][0], losses[1][0])
    assert losses[0][1]["num_pos"] == losses[1][1]["num_pos"]
