import torch

from discobax.models.pytorch_models import BayesianMLP


def make_model():
    model = BayesianMLP(input_size=4, hidden_size=32)
    model.eval()
    with torch.no_grad():
        model.fc1.weight.fill_(0.25)
        model.fc1.bias.fill_(0.5)
        model.fc2.weight.fill_(0.125)
        model.fc2.bias.zero_()
    return model


def test_candidates_share_each_mc_function_sample():
    model = make_model()
    identical_candidates = torch.ones((6, 4))
    torch.manual_seed(17)
    model.reset_mc_dropout_masks()

    samples = model([identical_candidates], k=12)[0]

    assert samples.shape == (6, 12)
    assert torch.equal(samples, samples[0].expand_as(samples))
    assert torch.unique(samples[0]).numel() > 1


def test_function_samples_do_not_depend_on_inference_batch_size():
    model = make_model()
    candidates = torch.arange(28, dtype=torch.float32).view(7, 4) / 10

    torch.manual_seed(29)
    model.reset_mc_dropout_masks()
    full = model([candidates], k=9)[0]

    torch.manual_seed(29)
    model.reset_mc_dropout_masks()
    chunks = torch.cat(
        [model([candidates[:3]], k=9)[0], model([candidates[3:]], k=9)[0]],
        dim=0,
    )

    # Matrix multiplication can differ at floating-point roundoff level when
    # its physical batch shape changes; the sampled functions/masks must agree.
    torch.testing.assert_close(full, chunks, rtol=1e-6, atol=1e-6)
