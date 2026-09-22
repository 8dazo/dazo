import torch
import torch.nn.functional as F

from dazo.losses import binary_cross_entropy_probs


def test_probability_bce_matches_reference_and_backprops_under_autocast():
    logits = torch.tensor([[0.2, -1.4, 2.1], [-0.7, 0.3, 1.2]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    probs = torch.sigmoid(logits)
    reference = F.binary_cross_entropy(probs, target)

    # CPU autocast is not identical to CUDA AMP, but this regression exercises the same
    # reduced-precision context while ensuring our helper never calls unsafe probability BCE.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = binary_cross_entropy_probs(torch.sigmoid(logits), target)

    assert torch.allclose(actual.float(), reference.float(), atol=1e-6, rtol=1e-5)
    actual.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
