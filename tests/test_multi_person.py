import numpy as np
import pytest
import torch

from unified_perception.config import Config, ModelConfig
from unified_perception.models.losses import SetCriterion
from unified_perception.models.sparse_head import SparseHead


def test_auxiliary_layer_average_does_not_change_loss_scale():
    criterion = SetCriterion(ModelConfig())
    scores = torch.zeros(1, 1, 1, 8, 1)
    predictions = torch.full((1, 1, 1, 8, 12, 3), 0.6)
    targets = torch.full((1, 1, 2, 12, 3), 0.5)
    valid = torch.ones(1, 1, 2, dtype=torch.bool)
    single, _ = criterion(scores, predictions, targets, valid)
    repeated, _ = criterion(scores.repeat(3, 1, 1, 1, 1), predictions.repeat(3, 1, 1, 1, 1, 1), targets, valid)
    torch.testing.assert_close(single, repeated)


@pytest.mark.parametrize("people", [0, 1, 2, 8])
def test_person_permutation_padding_and_gradients(people):
    torch.manual_seed(10)
    criterion = SetCriterion(ModelConfig())
    scores = torch.randn(2, 2, 3, 8, 1, requires_grad=True)
    predictions = torch.rand(2, 2, 3, 8, 12, 3, requires_grad=True)
    targets = torch.rand(2, 3, 8, 12, 3)
    valid = torch.zeros(2, 3, 8, dtype=torch.bool)
    valid[..., :people] = True
    loss, _ = criterion(scores, predictions, targets, valid)
    permutation = torch.randperm(8)
    permuted, _ = criterion(scores, predictions, targets[:, :, permutation], valid[:, :, permutation])
    torch.testing.assert_close(loss, permuted)
    loss.backward()
    assert torch.isfinite(scores.grad).all()
    if people:
        assert predictions.grad is not None and torch.isfinite(predictions.grad).all()


def test_query_overflow_is_not_silently_truncated():
    criterion = SetCriterion(ModelConfig())
    with pytest.raises(ValueError, match="queries"):
        criterion(torch.zeros(1, 1, 1, 1, 1), torch.zeros(1, 1, 1, 1, 12, 3),
                  torch.zeros(1, 1, 2, 12, 3), torch.ones(1, 1, 2, dtype=torch.bool))


@pytest.mark.parametrize("people", [0, 1, 2, 8])
def test_denoising_padding_and_gradients(people):
    criterion = SetCriterion(ModelConfig())
    valid = torch.zeros(1, 2, 8, dtype=torch.bool)
    valid[..., :people] = True
    cls = torch.zeros(2, 1, 2, 8, 1, requires_grad=True)
    kpts = torch.ones(2, 1, 2, 8, 12, 3, requires_grad=True)
    dn = {"dn_cls": cls, "dn_kpts": kpts, "target_kpts": torch.zeros(1, 2, 8, 12, 3),
          "valid": valid, "is_person": valid.clone()}
    loss_cls, loss_kpt = criterion._dn_loss(dn)
    loss = loss_cls + loss_kpt
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(cls.grad).all()
    if people:
        assert kpts.grad is not None and torch.isfinite(kpts.grad).all()


@pytest.mark.parametrize("people", [0, 1, 2, 8])
def test_sparse_head_multi_person_temporal_forward_backward_and_reset(people):
    torch.manual_seed(1)
    head = SparseHead(embed_dims=24, num_joints=12, num_query=8, num_propagated=2,
                      memory_len=8, topk_proposals=8, num_layers=2, num_heads=2,
                      ffn_dim=32, dropout=0., num_cams=2, num_levels=1, num_pts=1,
                      position_range=[-10, -10, 1, 10, 10, 20], scalar=2)
    feats = torch.randn(2, 3, 8, 24, requires_grad=True)
    codes = torch.randn(2, 3, 8, 14)
    projection = torch.eye(4).repeat(2, 3, 2, 1, 1)
    gt = torch.rand(2, 3, 8, 12, 3)
    valid = torch.zeros(2, 3, 8, dtype=torch.bool)
    valid[..., :people] = True
    valid[1, 1] = False
    cls, predictions, dn = head(feats, codes, projection, [(2, 2)], (16, 16), gt, valid)
    assert cls.shape == (2, 2, 3, 10, 1)
    loss, _ = SetCriterion(ModelConfig())(cls, predictions, gt, valid, dn)
    loss.backward()
    assert feats.grad is not None and torch.isfinite(feats.grad).all()
    head.eval()
    with torch.no_grad():
        first, _, _ = head(feats, codes, projection, [(2, 2)], (16, 16))
        second, _, _ = head(feats, codes, projection, [(2, 2)], (16, 16))
    torch.testing.assert_close(first, second)
