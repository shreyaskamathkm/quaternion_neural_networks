import torch

from quaternion_neural_networks.quaternion_ops import (
    get_i,
    get_j,
    get_k,
    get_r,
    hamilton_product,
    q_normalize,
)


def test_quaternion_ops_getters():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    assert torch.allclose(get_r(x), torch.tensor([[1.0], [5.0]]))
    assert torch.allclose(get_i(x), torch.tensor([[2.0], [6.0]]))
    assert torch.allclose(get_j(x), torch.tensor([[3.0], [7.0]]))
    assert torch.allclose(get_k(x), torch.tensor([[4.0], [8.0]]))

def test_q_normalize():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    norm_x = q_normalize(x)
    # The norm squared should be approximately 1
    r, i, j, k = get_r(norm_x), get_i(norm_x), get_j(norm_x), get_k(norm_x)
    assert torch.allclose(r*r + i*i + j*j + k*k, torch.tensor([[1.0]]), atol=1e-3)

def test_hamilton_product():
    # q1 = 1 + 2i + 3j + 4k
    q1 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    # q2 = 5 + 6i + 7j + 8k
    q2 = torch.tensor([[5.0, 6.0, 7.0, 8.0]])

    out = hamilton_product(q1, q2)
    # Expected result:
    # r = 1*5 - 2*6 - 3*7 - 4*8 = 5 - 12 - 21 - 32 = -60
    # i = 1*6 + 2*5 + 3*8 - 4*7 = 6 + 10 + 24 - 28 = 12
    # j = 1*7 - 2*8 + 3*5 + 4*6 = 7 - 16 + 15 + 24 = 30
    # k = 1*8 + 2*7 - 3*6 + 4*5 = 8 + 14 - 18 + 20 = 24

    expected = torch.tensor([[-60.0, 12.0, 30.0, 24.0]])
    assert torch.allclose(out, expected)
