# Basic Descriptor MLP

This experiment is the first fixed, interpretable model that consumes the
descriptor tables supplied by the doctoral team. It implements the reviewed
contract:

```text
normalized strain embedding (4096)
+ normalized chemical embedding (512)
+ medium one-hot (2)
+ perturbation time one-hot (6)
+ temperature one-hot (2)
-> 512 -> 256 latent -> 512 -> 4422 proteins
```

The four released validation scenarios are evaluated without using their
targets for preprocessing, early stopping, or model selection. Run outputs
are written under `runs/basic_descriptor_mlp/` at the repository root.

## Run

The project uses a separate environment. A typical setup is:

```bash
conda create -y -p experiments/basic_descriptor_mlp/.conda-env python=3.12 pip
experiments/basic_descriptor_mlp/.conda-env/bin/pip install -r experiments/basic_descriptor_mlp/requirements.txt
```

Then run the complete first wave on GPU 0:

```bash
experiments/basic_descriptor_mlp/.conda-env/bin/python \
  experiments/basic_descriptor_mlp/src/basic_descriptor_mlp/train.py \
  --config experiments/basic_descriptor_mlp/configs/base.yaml \
  --variant mean --device cpu

for variant in real zero shuffle; do
  experiments/basic_descriptor_mlp/.conda-env/bin/python \
    experiments/basic_descriptor_mlp/src/basic_descriptor_mlp/train.py \
    --config experiments/basic_descriptor_mlp/configs/base.yaml \
    --variant "$variant" --device cuda:0
done
```

`mean` is the metric-pipeline check. `real`, `zero`, and `shuffle` share every
training setting; only the descriptor information is changed. Each run saves
the config, hashes, feature contract, checkpoint, history, split metrics,
per-protein R2 summaries, and compressed validation predictions.

Descriptor provenance is deliberately marked as pending in
`provenance/descriptor_manifest.yaml`; no generation model or license is
inferred from the embedding dimensions.
