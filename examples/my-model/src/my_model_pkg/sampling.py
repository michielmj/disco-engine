import numpy as np
from numpy.random import Generator
import pandas as pd
from graphblas import Vector, op
from scipy.stats import distributions as scipy_dists


def sample_dists(
        rng: Generator,
        dists: list[tuple[any, np.ndarray]],  # distribution, indices
        sample_indices: np.ndarray,
        num_vertices: int,
        round_values: bool = False,
        lower: float = -float("inf"),
        upper: float = float("inf"),
) -> Vector:
    sample = Vector.from_coo(
        indices=sample_indices,
        values=np.zeros_like(sample_indices, dtype=np.dtypes.Float64DType),
        size=num_vertices
    )

    for dist, indices, params in dists:
        if sample_indices is None:
            needed = indices
            ixs = np.arange(indices.shape[0])
        else:
            needed = np.intersect1d(indices, sample_indices)
            ixs = np.searchsorted(indices, needed)

        if needed.shape[0] != 0:
            needed_params = params[:, ixs]

            values = dist.rvs(*needed_params, random_state=rng)
            values = np.asarray(values)
            if values.ndim == 0:
                values = np.asarray([values])

            if round_values:
                values = np.round(values)

            values[values < lower] = lower
            values[values > upper] = upper

            sample(op.plus) << Vector.from_coo(indices=needed, values=values, size=num_vertices)

    return sample.select('!=', 0).new()


def get_dists(data: pd.DataFrame, dists_column: str, params_column: str):
    dists = []

    for dist_name, dist_data in data.groupby(dists_column)[[dists_column, params_column]]:
        try:
            dist = getattr(scipy_dists, str(dist_name))
        except AttributeError:
            continue

        params = np.stack(dist_data[params_column].array).T
        dists.append((dist, dist_data.index.array, params))

    return dists
