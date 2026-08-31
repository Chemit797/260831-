from types import SimpleNamespace

import torch

from discobax.models.meta_models import PytorchMLPRegressorWithUncertainty
from discobax.models.pytorch_models import BayesianMLP


def make_wrapper(device="cpu"):
    inner = SimpleNamespace(model=BayesianMLP(input_size=4, hidden_size=3))
    inner.model.eval()
    return PytorchMLPRegressorWithUncertainty(inner, device=device)


def test_regular_samples_return_to_cpu_without_graph():
    wrapper = make_wrapper()
    output = wrapper.get_samples([torch.ones((2, 4))], k=2)[0]
    assert output.device.type == "cpu"
    assert not output.requires_grad


def test_adversarial_input_keeps_gradient_graph():
    wrapper = make_wrapper()
    values = torch.ones((1, 1, 4), requires_grad=True)
    output = wrapper.get_model_prediction(
        values, return_multiple_preds=True, num_target_samples=2
    )[0]
    output.var().backward()
    assert values.grad is not None
