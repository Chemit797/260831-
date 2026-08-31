import torch


VALID_DEVICE_NAMES = ("auto", "cpu", "cuda")


def resolve_device(device_name="auto"):
    """Resolve a portable device name without changing algorithm settings."""
    if device_name not in VALID_DEVICE_NAMES:
        raise ValueError(
            f"Unsupported device '{device_name}'. Expected one of {VALID_DEVICE_NAMES}."
        )
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_name)

