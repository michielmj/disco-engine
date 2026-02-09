import numpy as np

def sample_dists(
    random_state: np.random.RandomState,
    dists: list[tuple[any, np.ndarray, np.ndarray]],
    sample_indices: np.ndarray,
    round: bool = False,
    lower: float = -float("inf"),
    upper: float = float("inf"),
) -> Vector:
    sample = Vector.zeros(sample_indices)

    for dist, indices, params in dists:
        needed = Array.ind_isin(indices, sample_indices)
        needed_indices = indices[needed].copy()
        if needed_indices.shape[0] != 0:
            needed_params = params[:, needed]

            values = dist.rvs(*needed_params, random_state=random_state)
            values = np.asarray(values)
            if values.ndim == 0:
                values = np.asarray([values])

            if round:
                values = np.round(values)

            values[values < lower] = lower
            values[values > upper] = upper

            sample += Vector(indices=needed_indices, values=values)

    return sample


def get_dists(data: pd.DataFrame, dists_column: str, params_column: str):
    dists = []
    for dist_name in data[dists_column].unique():
        if dist_name is not None:
            try:
                dist = getattr(scipy_rv, dist_name)
            except:
                raise ValueError(f"Unsupported distribution type: {dist_name}.")
            else:
                df = data.loc[data[dists_column] == dist_name, [INDEX, params_column]]
                indices = df[INDEX].to_numpy(dtype=np.uint32)
                params = np.vstack(df[params_column]).T
                non_positive_means = dist.mean(*params) <= 0.0
                if np.any(non_positive_means):
                    raise SimulationRuntimeError(
                        f"There are distributions with non-positive means (e.g. with indices: "
                        f"{str(indices[non_positive_means][:10])})."
                    )

                dists.append((dist, indices, params))

    return dists