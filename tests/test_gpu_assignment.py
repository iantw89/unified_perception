"""The device-only track assignment (models/gpu_assignment.py) reproduces the host reference."""

import pytest
import torch

from unified_perception.config import LossConfig, ModelConfig
from unified_perception.models import gpu_assignment as ga
from unified_perception.models.losses import SetCriterion
from unified_perception.models.reid import DeviceReIDLoss, ReIDLoss
from unified_perception.models.supervision import Supervision, match_cost
from unified_perception.models.tracking import FALSE_TRACK_ID, TrackManager, TrackState

RANGE = [-1000.0, -1000.0, 0.0, 1000.0, 1000.0, 2000.0]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def scenario(seed, B=3, T=2, P=6, J=12, N=3, device="cpu"):
    """Random targets exercising every person category (located, root-only, 2-D-only, match-only)."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.rand(*s, generator=g)
    valid = r(B, T, P) < 0.7
    kind = (r(B, T, P) * 4).long()                         # 0 joints, 1 root only, 2 2-D only, 3 match only
    kv = (r(B, T, P, J) < 0.8) & (kind == 0)[..., None] & valid[..., None]
    root_valid = valid & (kind <= 1)
    gt = r(B, T, P, J, 3)
    gt = torch.where(kv[..., None], gt, torch.full_like(gt, float("nan")))      # golden: NaN when unlabeled
    root = torch.where(root_valid[..., None], r(B, T, P, 3), torch.full((B, T, P, 3), float("nan")))
    ids = torch.stack([torch.randperm(20, generator=g)[:P] for _ in range(B * T)]).view(B, T, P)
    ids = torch.where(r(B, T, P) < 0.1, -1, ids)
    ids[:, 1:] = ids[:, :1]                                # stable within the clip
    exists = valid | (r(B, T, P) < 0.3)
    kp2d = torch.cat([r(B, T, N, P, J, 2) * 640, r(B, T, N, P, J, 1)], -1)
    kp2d_valid = (r(B, T, N, P, J) < 0.6) & valid[:, :, None, :, None] & (kind == 2)[:, :, None, :, None]
    w2i = torch.eye(4).repeat(B, T, N, 1, 1)
    w2i[..., 0, 0] = w2i[..., 1, 1] = 0.3
    w2i[..., 0, 3] = w2i[..., 1, 3] = 320
    w2i[..., 2, 3] = 2500
    match_root = r(B, T, P, 3)
    match_root_valid = valid & (kind == 3)
    targets = {"gt": gt, "valid": valid, "keypoint_valid": kv, "frame_valid": r(B, T) < 0.7, "root": root,
               "root_valid": root_valid, "instance_ids": ids, "exists": exists, "kp2d": kp2d,
               "kp2d_valid": kp2d_valid, "world2img": w2i, "match_root": match_root,
               "match_root_valid": match_root_valid, "sigma_mm": r(B, T, P, J) * 50,
               "root_sigma_mm": r(B, T, P) * 50}
    return {k: v.to(device) for k, v in targets.items()}


def outputs(seed, B, num_det, K, J=12, C=8, D=4, device="cpu"):
    g = torch.Generator().manual_seed(seed + 1000)
    Qm = num_det + K
    return (torch.randn(B, Qm, 1, generator=g).to(device), torch.rand(B, Qm, J, 3, generator=g).to(device),
            torch.randn(B, Qm, C, generator=g).to(device), torch.nn.functional.normalize(
                torch.randn(B, Qm, D, generator=g), dim=-1).to(device),
            torch.nn.functional.normalize(torch.randn(B, K, D, generator=g), dim=-1).to(device))


def manager(device="cpu", **overrides):
    cfg = ModelConfig(track_queries=True, track_drop_prob=0.0, track_fp_prob=0.0, track_ref_noise=0.0,
                      reid_dims=4)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return TrackManager(cfg, RANGE)


def random_state(seed, B, K, ids, C=8, D=4, device="cpu"):
    g = torch.Generator().manual_seed(seed + 2000)
    state = TrackState.empty(B, K, C, device, torch.float32, reid_dims=D, gallery=ModelConfig().reid_gallery_size)
    state.valid.copy_(torch.rand(B, K, generator=g) < 0.6)
    pool = torch.cat([ids[:, 0], torch.full((B, 4), 99)], dim=1)                # present ids + gone ones
    pick = (torch.rand(B, K, generator=g) * pool.shape[1]).long()
    state.ids.copy_(torch.where(state.valid, torch.gather(pool.cpu(), 1, pick).to(device), -1))
    state.embed.copy_(torch.randn(B, K, C, generator=g))
    state.ref.copy_(torch.rand(B, K, 3, generator=g))
    state.inactive.copy_((torch.rand(B, K, generator=g) * 3).long())
    state.id_ema.copy_(torch.nn.functional.normalize(torch.randn(B, K, D, generator=g), dim=-1))
    return state


def legacy_tensors(assign, Qm):
    target = torch.full((Qm,), -1, dtype=torch.long)
    excluded = torch.zeros(Qm, dtype=torch.bool)
    for q, g in zip(assign["track_q"] + assign["det_q"], assign["track_g"] + assign["det_g"]):
        target[q] = g
    for q in assign["ignore"] + assign["invalid"]:
        excluded[q] = True
    return target, excluded


@pytest.mark.parametrize("seed", range(6))
def test_batch_cost_matches_per_lane_cost(seed):
    t = scenario(seed)
    sup = Supervision(t, torch.tensor(RANGE))
    tm = manager()
    cls, kpts, *_ = outputs(seed, 3, 10, 0)
    cost = ga.batch_match_cost(sup, 1, cls, kpts, tm.costs)
    for b in range(3):
        ref, people = match_cost(sup.frame(b, 1), cls[b], kpts[b], list(range(6)), tm.costs)
        if ref is not None:
            torch.testing.assert_close(cost[b][:, people], ref.float(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_solver_matches_scipy(device):
    from scipy.optimize import linear_sum_assignment
    g = torch.Generator().manual_seed(0)
    for P in (1, 5, 24):
        cost = torch.rand(4, 64, P, generator=g)
        available = torch.rand(4, P, generator=g) < 0.7
        out = ga.solve_assignment(cost.to(device), available.to(device)).cpu()
        for b in range(4):
            people = available[b].nonzero().flatten()
            expected = torch.full((64,), -1, dtype=torch.long)
            if len(people):
                r, c = linear_sum_assignment(cost[b][:, people].numpy())
                expected[torch.as_tensor(r)] = people[torch.as_tensor(c)]
            assert torch.equal(out[b], expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", range(8))
def test_train_step_matches_reference(device, seed):
    B, P, K, num_det = 3, 6, 5, 10
    t_cpu = scenario(seed, B=B, P=P)
    tm_ref, tm_gpu = manager(), manager(gpu_assignment=True)
    cls, kpts, feat, codes, ema = outputs(seed, B, num_det, K)
    state = random_state(seed, B, K, t_cpu["instance_ids"])
    ref_assign, ref_new = tm_ref.train_step(state, num_det, cls, kpts, feat, dict(t_cpu), 1, codes=codes, ema=ema)

    t_dev = scenario(seed, B=B, P=P, device=device)
    state_dev = random_state(seed, B, K, t_cpu["instance_ids"], device=device)
    move = lambda x: x.to(device)
    assign, new = tm_gpu.train_step(state_dev, num_det, move(cls), move(kpts), move(feat), t_dev, 1,
                                    codes=move(codes), ema=move(ema))
    for b in range(B):
        target, excluded = legacy_tensors(ref_assign[b], num_det + K)
        assert torch.equal(assign["target"][b].cpu(), target)
        assert torch.equal(assign["excluded"][b].cpu(), excluded)
    for name in ("valid", "ids", "inactive"):
        assert torch.equal(getattr(new, name).cpu(), getattr(ref_new, name)), name
    for name in ("embed", "ref", "id_ema"):
        torch.testing.assert_close(getattr(new, name).cpu(), getattr(ref_new, name), msg=name)


def test_drop_and_false_track_rules():
    B, P, K, num_det = 2, 4, 3, 6
    t = scenario(3, B=B, P=P)
    cls, kpts, feat, codes, ema = outputs(3, B, num_det, K)
    state = random_state(3, B, K, t["instance_ids"])
    tm = manager(gpu_assignment=True, track_drop_prob=1.0, track_fp_prob=1.0)
    assign, new = tm.train_step(state, num_det, cls, kpts, feat, t, 1, codes=codes, ema=ema)
    assert new.ids[:, 0].tolist() == [FALSE_TRACK_ID] * B and not new.valid[:, 1:].any()
    matched = assign["target"][:, :num_det] >= 0
    best = cls[:, :num_det, 0].masked_fill(matched, float("-inf")).argmax(1)
    torch.testing.assert_close(new.embed[:, 0], feat[torch.arange(B), best])


def test_device_reid_loss_matches_reference():
    torch.manual_seed(0)
    B, T, n_q, n_s, D = 2, 4, 5, 2, 8
    ref = ReIDLoss(0.1, recent=3, anchors=2, stride=2)
    dev = DeviceReIDLoss(0.1, recent=3, anchors=2, stride=2)
    for window in range(5):
        codes = torch.nn.functional.normalize(torch.randn(B, T * (n_q + n_s), D), dim=-1).requires_grad_()
        labels = torch.randint(0, 4, (B, T * (n_q + n_s)))
        frames = torch.cat([torch.arange(T).repeat_interleave(n_q), torch.arange(T).repeat_interleave(n_s)])
        # one entry per identity per frame block
        for b in range(B):
            for f in range(T):
                for lo, width in ((f * n_q, n_q), (T * n_q + f * n_s, n_s)):
                    labels[b, lo:lo + width] = torch.randperm(6)[:width]
        mask = torch.rand(B, T * (n_q + n_s)) < 0.7
        reset = torch.tensor([window == 0, window in (0, 3)])
        groups = [torch.cat([frames[:T * n_q] == f, torch.zeros(T * n_s, dtype=torch.bool)]) for f in range(T)]
        groups += [torch.cat([torch.zeros(T * n_q, dtype=torch.bool), frames[T * n_q:] == f]) for f in range(T)]
        samples = [(codes[b][mask[b]], labels[b][mask[b]], frames[mask[b]]) for b in range(B)]
        expected, n_expected = ref(samples, reset, T)
        loss, count = dev(codes, labels, frames.expand(B, -1), mask, groups, reset, T)
        assert int(count) == n_expected
        if n_expected:
            torch.testing.assert_close(loss, expected)


def test_device_loss_matches_shared_reference():
    torch.manual_seed(0)
    cfg = ModelConfig(reid_dims=0)
    criterion = SetCriterion(cfg, LossConfig(share_aux_matching=True, root_weight=5.0))
    L, B, T, Qm, P, num_det = 2, 2, 3, 7, 4, 4
    t = scenario(5, B=B, T=T, P=P)
    cls = torch.randn(L, B, T, Qm, 1, requires_grad=True)
    kpts = torch.rand(L, B, T, Qm, 12, 3, requires_grad=True)
    target = torch.full((B, T, Qm), -1, dtype=torch.long)
    excluded = torch.zeros(B, T, Qm, dtype=torch.bool)
    lists = [[None] * B for _ in range(T)]
    for b in range(B):
        for f in range(T):
            people = t["valid"][b, f].nonzero().flatten().tolist()
            qs = torch.randperm(Qm)[:len(people)].tolist()
            for q, g in zip(qs, people):
                target[b, f, q] = g
            skip = [q for q in range(num_det, Qm) if q not in qs][:1]
            excluded[b, f, skip] = True
            lists[f][b] = {"track_q": [q for q in qs if q >= num_det], "track_g": [g for q, g in zip(qs, people) if q >= num_det],
                           "det_q": [q for q in qs if q < num_det], "det_g": [g for q, g in zip(qs, people) if q < num_det],
                           "background": [], "ignore": skip, "invalid": []}
    common = dict(gt_keypoint_frame_valid=t["frame_valid"], gt_root=t["root"], gt_root_valid=t["root_valid"],
                  gt_keypoint_valid=t["keypoint_valid"], gt_keypoint_sigma=t["sigma_mm"],
                  gt_root_sigma=t["root_sigma_mm"], gt_kp2d=t["kp2d"], gt_kp2d_valid=t["kp2d_valid"],
                  world2img=t["world2img"], position_range=torch.tensor(RANGE))
    ref_loss, ref_metrics = criterion(cls, kpts, t["gt"], t["valid"], track_output={"num_det": num_det,
                                      "assignments": lists}, **common)
    loss, metrics = criterion(cls, kpts, t["gt"], t["valid"], track_output={"num_det": num_det,
                              "assignments": {"target": target, "excluded": excluded}}, **common)
    torch.testing.assert_close(loss, ref_loss)
    for key in ("loss_cls", "loss_kpt", "loss_root", "num_pos", "num_kpt", "num_root", "num_track_pairs"):
        assert float(metrics[key]) == pytest.approx(float(ref_metrics[key]), rel=1e-5), key


@pytest.mark.parametrize("seed", range(4))
def test_train_step_matches_reference_with_crop_appearance(seed):
    B, P, K, num_det, E = 3, 6, 5, 10, 6
    t = scenario(seed, B=B, P=P)
    cls, kpts, feat, codes, ema = outputs(seed, B, num_det, K)
    g = torch.Generator().manual_seed(seed + 3000)
    app = (torch.randn(B, K, E, generator=g), torch.rand(B, K, generator=g) < 0.5)

    def fresh():
        state = random_state(seed, B, K, t["instance_ids"])
        full = TrackState.empty(B, K, 8, "cpu", torch.float32, reid_dims=4,
                                gallery=ModelConfig().reid_gallery_size, app_dims=E)
        for name in ("valid", "ids", "embed", "ref", "inactive", "id_ema"):
            getattr(full, name).copy_(getattr(state, name))
        full.app_valid.copy_(torch.rand(B, K, generator=torch.Generator().manual_seed(seed)) < 0.5)
        full.app_ema.copy_(torch.randn(B, K, E, generator=torch.Generator().manual_seed(seed)))
        return full

    _, ref_new = manager().train_step(fresh(), num_det, cls, kpts, feat, dict(t), 1, codes=codes, ema=ema, app=app)
    _, new = manager(gpu_assignment=True).train_step(fresh(), num_det, cls, kpts, feat, dict(t), 1,
                                                     codes=codes, ema=ema, app=app)
    assert torch.equal(new.app_valid, ref_new.app_valid)
    torch.testing.assert_close(new.app_ema, ref_new.app_ema)


@pytest.mark.parametrize("device", DEVICES)
def test_solver_survives_nonfinite_costs(device):
    cost = torch.rand(3, 16, 5)
    cost[0, :, 2] = float("nan")          # a person whose whole cost column is NaN
    cost[1, 4, :] = float("inf")          # a query with infinite cost to everyone
    available = torch.ones(3, 5, dtype=torch.bool)
    out = ga.solve_assignment(cost.to(device), available.to(device))
    if device == "cuda":
        torch.cuda.synchronize()          # an out-of-range scatter would surface here
    out = out.cpu()
    assert ((out >= -1) & (out < 5)).all()
    for b in range(3):                    # each person matched at most once
        matched = out[b][out[b] >= 0]
        assert len(matched) == len(set(matched.tolist()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA autocast")
def test_fused_trunk_built_under_autocast_stays_fp32():
    from unified_perception.config import load_config
    from unified_perception.models import UnifiedPerceptionModel
    cfg = load_config("configs/gym_sixcam_track_v1.yaml")
    model = UnifiedPerceptionModel(cfg).cuda()
    bb = model.backbone
    x = torch.rand(2, 3, 640, 640, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        bb._forward(x)                    # first build happens inside the training autocast
    assert all(p.dtype == torch.float32 for p in bb._fused_trunk().parameters())
    with torch.no_grad():
        bb._forward(x)                    # evaluation path: no autocast, fp32 input
