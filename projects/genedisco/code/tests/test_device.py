import pytest
import torch

from discobax.utils.device import resolve_device


def test_resolve_cpu():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_auto():
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("auto") == torch.device(expected)


def test_reject_unknown_device():
    with pytest.raises(ValueError, match="Unsupported device"):
        resolve_device("tpu")

