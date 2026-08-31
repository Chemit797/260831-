import argparse
from pathlib import Path

import h5py
import numpy as np


FILES = [
    "achilles.h5",
    "schmidt_2021_ifng.h5",
    "schmidt_2021_il2.h5",
    "zhuang_2019.h5",
    "sanchez_2021_neurons_tau.h5",
    "zhu_2021_sarscov2_host_factors.h5",
]


def decode(values):
    return np.asarray(
        [value.decode() if isinstance(value, bytes) else value for value in values]
    )


def main():
    parser = argparse.ArgumentParser(description="Compare two GeneDisco HDF5 caches.")
    parser.add_argument("release", type=Path)
    parser.add_argument("other", type=Path)
    args = parser.parse_args()

    for name in FILES:
        with h5py.File(args.release / name, "r") as release_file, h5py.File(
            args.other / name, "r"
        ) as other_file:
            release_rows = decode(release_file["rownames"][...])
            other_rows = decode(other_file["rownames"][...])
            common = np.intersect1d(release_rows, other_rows)
            release_index = {value: index for index, value in enumerate(release_rows)}
            other_index = {value: index for index, value in enumerate(other_rows)}
            release_values = release_file["covariates"][
                [release_index[value] for value in common]
            ].astype(float)
            other_values = other_file["covariates"][
                [other_index[value] for value in common]
            ].astype(float)
            difference = np.abs(release_values - other_values)
            print(
                name,
                f"rows={len(release_rows)}/{len(other_rows)}",
                f"common={len(common)}",
                f"mean_abs={np.nanmean(difference):.8g}",
                f"max_abs={np.nanmax(difference):.8g}",
            )


if __name__ == "__main__":
    main()
