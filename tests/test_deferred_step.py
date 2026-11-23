"""TrainStep.step_deferred: no host sync, and a non-finite step is skipped on the device."""

import pytest
import torch

from unified_perception.engine.step import TrainStep

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="deferred steps need CUDA")


class _Head(torch.nn.Module):
    track_output = None


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 1)
        self.head = _Head()
        self.position_range = torch.zeros(6)


def _step(poison: bool):
    model = _Model().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=0.1, fused=True)
    ts = TrainStep(model, None, opt, None, None, torch.device("cuda"), False)
    assert ts.deferred_supported()
    x = torch.randn(4, 3, device="cuda")

    def criterion_outputs():
        loss = model.lin(x).pow(2).mean()
        if poison:
            loss = loss * float("nan")
        return loss, {"loss_kpt": loss.detach(), "num_pos": torch.full((), 4, device="cuda")}

    ts._criterion_outputs = criterion_outputs
    ts.free_inputs = lambda: None
    before = [p.detach().clone() for p in model.parameters()]
    torch.cuda.set_sync_debug_mode("error")       # any host synchronization raises
    try:
        pending = ts.step_deferred(1.0)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    result = pending.result()
    after = list(model.parameters())
    steps = [float(opt.state[p]["step"]) for p in model.parameters()]
    return result, before, after, steps


def test_finite_step_updates_without_sync():
    result, before, after, steps = _step(poison=False)
    assert result["skipped"] == 0.0 and result["num_pos"] == 4
    assert any(not torch.equal(b, a) for b, a in zip(before, after))
    assert steps == [1.0, 1.0]


def test_nonfinite_step_is_skipped_on_device():
    result, before, after, steps = _step(poison=True)
    assert result["skipped"] == 1.0
    assert all(torch.equal(b, a) for b, a in zip(before, after))
    assert steps == [0.0, 0.0]                    # the step counter is rolled back too
