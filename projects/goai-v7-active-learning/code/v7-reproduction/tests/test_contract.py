from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "experiments" / "proteome_biostate_readout_v7_reproduction" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from proteome_biostate_readout.model import ProteomeBioStateReadout, masked_mse  # noqa: E402
from proteome_biostate_readout.train import SideState  # noqa: E402


class ContractTests(unittest.TestCase):
    def test_side_vocab_is_train_only_and_unknown_is_neutral(self) -> None:
        metadata = pd.DataFrame(
            {
                "instrument": ["train_instr", "validation_only"],
                "Yeast_cell_plate": ["train_plate", "validation_plate"],
            },
            index=["train", "validation"],
        )
        state = SideState.fit(metadata, pd.Index(["train"]))
        instrument, plate = state.transform(metadata.loc[["validation"]])
        np.testing.assert_array_equal(instrument, [0])
        np.testing.assert_array_equal(plate, [0])

    def test_forward_shapes_and_condition_only_prediction(self) -> None:
        model = ProteomeBioStateReadout(
            strain_dim=4,
            chemical_dim=3,
            context_dim=2,
            n_instruments=3,
            n_plates=4,
            n_proteins=5,
            hidden_dim=16,
            posterior_hidden_dim=16,
            decoder_hidden_dim=16,
            latent_dim=8,
            side_embedding_dim=3,
            norm="layernorm",
        )
        condition = torch.randn(6, 9)
        target = torch.randn(6, 5)
        mask = torch.rand(6, 5) > 0.2
        instrument = torch.tensor([0, 1, 2, 1, 2, 0])
        plate = torch.tensor([0, 1, 2, 3, 1, 0])
        output = model(condition, instrument, plate, target, mask, True, True, 1.0)
        self.assertEqual(tuple(output["pred_c"].shape), (6, 5))
        self.assertEqual(tuple(output["posterior_mu"].shape), (6, 8))
        prediction, mask_logits = model.predict_from_condition(condition, instrument, plate)
        self.assertEqual(tuple(prediction.shape), (6, 5))
        self.assertEqual(tuple(mask_logits.shape), (6, 5))

    def test_masked_mse_ignores_unobserved_values(self) -> None:
        prediction = torch.tensor([[2.0, 100.0]])
        target = torch.tensor([[1.0, -100.0]])
        mask = torch.tensor([[True, False]])
        self.assertAlmostEqual(float(masked_mse(prediction, target, mask)), 1.0)


if __name__ == "__main__":
    unittest.main()

