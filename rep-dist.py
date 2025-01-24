import numpy as np
import numpy.typing as npt
import plotly.colors as co
import plotly.figure_factory as ff
import torch


def get_dist(conditional: npt.NDArray) -> npt.NDArray:
    one_hot = (conditional == conditional.max(axis=1)[:, None]).astype(int)
    counts = np.sum(one_hot, axis=0)
    norm = np.sum(counts)
    # return counts / np.full(counts.shape, norm)
    return counts


def gen_hist(data: npt.NDArray):
    return [
        epitope_id
        for epitope_id in range(data.shape[0])
        for _ in range(data[epitope_id])
    ]


if __name__ == "__main__":
    files = [
        "./ignore/dcr_LTX_0001_N_beta_all_probability.pt",
        "./ignore/dcr_LTX_0001_PBMCFU_beta_all_probability.pt",
        "./ignore/dcr_LTX_0001_R1_beta_all_probability.pt",
    ]

    hists = []
    for file in files:
        data = torch.load(file, weights_only=True).numpy()
        dist = get_dist(data)
        hist = gen_hist(dist)
        hists.append(hist)

    colors = co.qualitative.Plotly[:3]

    fig = ff.create_distplot(
        hists,
        ["Blood", "Tumour tissue", "Non-tumour tissue"],
        colors=colors,
        show_rug=False,
    )

    fig.for_each_trace(lambda trace: trace.update(opacity=0.65))

    fig.update_layout(
        xaxis=dict(title=dict(text="Epitope Number")),
        yaxis=dict(title=dict(text="Predicted Frequency")),
    )

    fig.write_image("./ignore/ep_dist_plot.png", scale=5)
