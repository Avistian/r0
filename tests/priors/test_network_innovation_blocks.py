"""Public block slices preserve nonlinear parent ordering and projected skips."""

import numpy as np
import torch

from r0.modules import priors as p


def test_network_block_uses_ordered_slices_and_skip_oracle():
    block = p.NetworkBlock(
        (1, 0),
        (
            p.NetworkAffine(
                ((2.0, -3.0, 4.0),),
                (0.5,),
                activation=p.CausalActivation("tanh"),
                placement="before",
            ),
        ),
        skips=(p.NetworkProjection(0, p.NetworkAffine(((7.0, -2.0),), (1.0,))),),
    )
    function = p.FrozenNetworkBlockFunction((2, 1), block)
    x = torch.tensor([[0.2, -0.7, 1.1], [2.0, 3.0, -0.4]], dtype=torch.float64)
    expected = np.tanh(x.numpy()[:, [2, 0, 1]]) @ np.array([2.0, -3.0, 4.0]) + 0.5
    expected += x.numpy()[:, :2] @ np.array([7.0, -2.0]) + 1.0
    actual = p.evaluate_frozen_function(function, x)
    np.testing.assert_allclose(actual.numpy()[:, 0], expected, rtol=1e-14)
