import torch

from dazo.core import DazoCore


def build():
    torch.manual_seed(3)
    return DazoCore(
        context_dim=64,
        option_dim=48,
        latent_dim=32,
        n_evidence_slots=4,
        n_hypothesis_slots=3,
        n_critic_slots=2,
        n_control_slots=1,
        layers_per_loop=2,
        heads=4,
        dropout=0.0,
        max_loop_embeddings=8,
    ).eval()


def test_shapes_and_masks():
    model = build()
    context = torch.randn(2, 11, 64)
    context_mask = torch.ones(2, 11, dtype=torch.bool)
    options = torch.randn(2, 5, 48)
    option_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    out = model(context, context_mask, options, option_mask, max_steps=4)
    assert out.logits.shape == (2, 5)
    assert out.per_step_logits.shape == (2, 4, 5)
    assert out.halt_probs.shape == (2, 4)
    assert out.convergence.shape == (2, 4)
    assert torch.allclose(out.probs.sum(-1), torch.ones(2), atol=1e-5)
    assert torch.all(out.probs[1, 3:] < 1e-7)


def test_option_permutation_equivariance():
    model = build()
    context = torch.randn(2, 9, 64)
    context_mask = torch.ones(2, 9, dtype=torch.bool)
    options = torch.randn(2, 5, 48)
    mask = torch.ones(2, 5, dtype=torch.bool)
    ranks = torch.zeros(2, 5, dtype=torch.long)
    out = model(context, context_mask, options, mask, rank_ids=ranks, max_steps=3)
    perm = torch.tensor([2, 4, 0, 3, 1])
    inv = torch.argsort(perm)
    out_p = model(context, context_mask, options[:, perm], mask[:, perm], rank_ids=ranks[:, perm], max_steps=3)
    assert torch.allclose(out.logits, out_p.logits[:, inv], atol=1e-5, rtol=1e-5)


def test_ordinal_rank_moves_with_option():
    model = build()
    context = torch.randn(1, 7, 64)
    context_mask = torch.ones(1, 7, dtype=torch.bool)
    options = torch.randn(1, 4, 48)
    mask = torch.ones(1, 4, dtype=torch.bool)
    ranks = torch.tensor([[1, 2, 3, 4]])
    out = model(context, context_mask, options, mask, rank_ids=ranks, max_steps=2)
    perm = torch.tensor([3, 1, 0, 2])
    inv = torch.argsort(perm)
    out_p = model(context, context_mask, options[:, perm], mask[:, perm], rank_ids=ranks[:, perm], max_steps=2)
    assert torch.allclose(out.logits, out_p.logits[:, inv], atol=1e-5, rtol=1e-5)


def test_backward():
    model = build().train()
    context = torch.randn(2, 8, 64)
    options = torch.randn(2, 3, 48)
    mask_c = torch.ones(2, 8, dtype=torch.bool)
    mask_o = torch.ones(2, 3, dtype=torch.bool)
    out = model(context, mask_c, options, mask_o, max_steps=2)
    loss = out.logits.mean() + out.halt_probs.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
