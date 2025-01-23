import plotly.graph_objects as go
import numpy as np
import torch
import torch.nn.functional as F

if __name__ == "__main__":
    data = torch.load("./ignore/all_probability.pt", weights_only=True)
    data = data.numpy()
    one_hot = (data == data.max(axis=1)[:,None]).astype(int)
    counts = np.sum(one_hot, axis=0)
    print(counts)

    # fig = go.Figure()
    #
    # fig.add_trace(
    #     go.
    # )
