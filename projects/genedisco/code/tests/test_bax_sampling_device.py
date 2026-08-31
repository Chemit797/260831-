import numpy as np
import pytest
import torch

from discobax.methods.bax_acquisition.bax_sampling import (
    BaxAcquisition,
    bernoulli_noise_sampler,
    build_noise_kernel,
    gaussian_noise_sampler,
    noise_subset_select,
    subset_select,
)


class TinyDataSource:
    def __init__(self, values, row_names=None):
        self.values = np.asarray(values, dtype=np.float32)
        self.row_names = list(range(len(self.values))) if row_names is None else list(row_names)

    def subset(self, indices):
        positions = [self.row_names.index(index) for index in indices]
        return TinyDataSource(self.values[positions], indices)

    def get_data(self):
        return [self.values]

    def __len__(self):
        return len(self.values)


class RestoreTrackingModel:
    prediction_states = []

    def __init__(self, state=0):
        self.state = state

    def save_folder(self, _path):
        return None

    def load_folder(self, _path):
        return RestoreTrackingModel(state=0)

    def fit(self, _x, _y):
        self.state = 1
        return self

    def get_model_prediction(
        self, data, return_multiple_preds=False, num_target_samples=None
    ):
        self.prediction_states.append(self.state)
        count = len(data)
        if return_multiple_preds:
            values = torch.arange(count, dtype=torch.float32).view(-1, 1)
            return [values.repeat(1, num_target_samples)]
        return [torch.arange(count, dtype=torch.float32)]


@pytest.mark.parametrize("sampler", [gaussian_noise_sampler, bernoulli_noise_sampler])
def test_noise_sampler_cpu_shape_and_floor(sampler):
    x = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(-1, 1)
    fx = np.linspace(-0.5, 0.5, 8, dtype=np.float32)
    sampled = sampler(x, fx, seed=7, device="cpu")
    assert sampled.shape == fx.shape
    assert np.all(sampled >= 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_gaussian_noise_sampler_cuda():
    x = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(-1, 1)
    fx = np.linspace(-0.5, 0.5, 8, dtype=np.float32)
    sampled = gaussian_noise_sampler(x, fx, seed=7, device="cuda")
    assert sampled.shape == fx.shape
    assert np.all(sampled >= 0)


def test_subset_select_budget_is_configurable():
    calls = []

    def sampler(values):
        calls.append(1)
        return np.maximum(values, 0)

    result = subset_select(np.array([0.1, 0.5, 0.2]), sampler, 2, budget=3)
    assert len(calls) == 3
    assert len(result) == 2


def test_noise_subset_select_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unsupported noise type"):
        noise_subset_select("unknown", np.zeros((2, 1)))


def test_noise_kernel_uses_requested_constrained_values():
    kernel = build_noise_kernel(1.25, 2.5, device="cpu")

    assert kernel.base_kernel.lengthscale.item() == pytest.approx(1.25)
    assert kernel.outputscale.item() == pytest.approx(2.5)


def test_each_bax_conditional_fit_restores_the_baseline(tmp_path):
    RestoreTrackingModel.prediction_states = []
    dataset_x = TinyDataSource(np.arange(4).reshape(-1, 1))
    dataset_y = TinyDataSource(np.arange(4).reshape(-1, 1))
    acquisition = BaxAcquisition(
        objective_function="topk",
        k=1,
        num_samples_EIG=2,
        num_samples_entropy=2,
        device="cpu",
    )

    proposal = acquisition(
        dataset_x=dataset_x,
        acquisition_batch_size=1,
        available_indices=[1, 2, 3],
        last_selected_indices=[0],
        cumulative_indices=[0],
        model=RestoreTrackingModel(),
        dataset_y=dataset_y,
        temp_folder_name=str(tmp_path / "baseline"),
    )

    assert len(proposal) == 1
    # Each branch samples at state 0, then evaluates its fitted conditional at
    # state 1.  The final unconditional entropy is restored to state 0.
    assert RestoreTrackingModel.prediction_states == [0, 1, 0, 1, 0]
