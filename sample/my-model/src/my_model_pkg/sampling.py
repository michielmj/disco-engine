import numpy as np
import pandas as pd
from graphblas import Vector, op
from toolbox import distributions


def sample_dists(
        random_state: np.random.RandomState,
        dists: list[tuple[any, np.ndarray, np.ndarray]],
        sample_indices: np.ndarray,
        round_values: bool = False,
        lower: float = -float("inf"),
        upper: float = float("inf"),
) -> Vector:
    sample = Vector.from_coo(indices=sample_indices, values=np.zeros_like(sample_indices, dtype=np.dtypes.Float64DType))

    for dist, indices, params in dists:

        needed = np.intersect1d(indices, sample_indices)
        if needed.shape[0] != 0:
            needed_params = params[:, needed]

            values = dist.rvs(*needed_params, random_state=random_state)
            values = np.asarray(values)
            if values.ndim == 0:
                values = np.asarray([values])

            if round_values:
                values = np.round(values)

            values[values < lower] = lower
            values[values > upper] = upper

            sample(op.plus) << Vector.from_coo(indices=needed, values=values)

    return sample


def get_dists(data: pd.DataFrame, dists_column: str, params_column: str):
    dists = []

    data = data.dropna(subset=[dists_column])
    for dist_name in data[dists_column].unique():
        try:
            dist = getattr(distributions, dist_name)
        except:
            raise ValueError(f"Unsupported distribution type: {dist_name}.")
        else:
            df = data.loc[data[dists_column] == dist_name, [params_column]]
            indices = df.index.to_numpy(dtype=np.int64)
            params = np.vstack(df[params_column]).T
            non_positive_means = dist.mean(*params) <= 0.0
            if np.any(non_positive_means):
                raise ValueError(
                    f"There are distributions with non-positive means (e.g. with indices: "
                    f"{str(indices[non_positive_means][:10])})."
                )

            dists.append((dist, indices, params))

    return dists
