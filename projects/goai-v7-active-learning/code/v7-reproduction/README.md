# Proteome BioState-Readout v7 Reproduction

This experiment adapts only the proteome branch of
`WAYFM_train_biostate_readout_v7_final.py` to the released GOAI WAYB/WAYC
train/validation contract. It does not enable metabolome, growth, KO routing,
or multi-omics alignment.

## Comparison contract

- Same metadata, proteome table, train rows, four released validation splits,
  retained protein panel, descriptors, seeds 42/43/44, and 100-epoch budget as
  `basic_descriptor_mlp`.
- The single primary change is the model/loss recipe: routed condition prior,
  proteome posterior, shared observer, mask head, KL/NCE alignment, and
  instrument adversary.
- Target filtering and normalization are fitted on `split_final == train` only.
- Instrument and plate vocabularies are fitted on training rows only. Validation
  categories unseen during training map to a fixed neutral `<UNK>` embedding.
- Released validation targets are used only for final reporting, never for
  preprocessing, early stopping, or checkpoint selection.

## Architecture adaptation

The GOAI table contains chemical perturbations and no KO field, so the routed
encoder uses the chemical route only:

```text
strain expert + chemical expert + medium/time/temperature context expert
    -> z_bio -> z_pro_prior -> proteome observer -> condition prediction

observed proteome + observation mask
    -> z_pro_posterior -> same observer -> posterior reconstruction
```

The observer receives train-fitted instrument and plate indices. Training uses
the v7 Stage-1 losses: condition prediction, posterior reconstruction, mask
prediction, posterior-to-prior KL, prior/posterior NCE, bio/posterior NCE, and
instrument adversarial loss.

## Run

```bash
PYTHONPATH=experiments/proteome_biostate_readout_v7_reproduction/src \
experiments/basic_descriptor_mlp/.conda-env/bin/python \
  experiments/proteome_biostate_readout_v7_reproduction/src/proteome_biostate_readout/train.py \
  --config experiments/proteome_biostate_readout_v7_reproduction/configs/proteome_only.yaml \
  --device cuda:0 --seed 42
```

Repeat for seeds 43 and 44. Build the common-mask comparison table with
`scripts/build_comparison_metrics.py`, then render the PNG/SVG with
`scripts/plot_comparison.py`.

