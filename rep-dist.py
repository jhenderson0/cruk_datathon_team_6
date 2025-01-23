import plotly.graph_objects as go
import numpy as np
import torch
import torch.nn.functional as F
import numpy.typing as npt

def get_dist(conditional: npt.NDArray) -> npt.NDArray:
    one_hot = (conditional == conditional.max(axis=1)[:,None]).astype(int)
    counts = np.sum(one_hot, axis=0)
    norm = np.sum(counts)
    return counts / np.full(counts.shape, norm)


if __name__ == "__main__":
    normal = torch.load("./ignore/all_probability.pt", weights_only=True).numpy()
    norm_dist = get_dist(normal)
    print(norm_dist)

    # fig = go.Figure()
    #
    # fig.add_trace(
    #     go.
    # )
