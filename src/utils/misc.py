import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def maybe_split(dir_name: str) -> str:
    """
    Recursively splits a given dir_name at half, once it exceeds max folder size of 255.
    """
    if len(dir_name) > 255:
        half = len(dir_name) // 2
        dir_name = maybe_split(dir_name[:half]) + "/" + maybe_split(dir_name[half:])
    return dir_name


def load_layer_stats(path, layer_idx=0, masked=False):
    """
    Load layer stats from a given path. Assumes the file is a .json file.
    Args:
        path: Str. Path to stats file.
        layer_idx: Int. Index of layer stats to load.

    """
    with open(path, "r") as f:
        stats = json.load(f)
    if layer_idx < 0:
        layer_idx = list(stats.keys())[layer_idx]
    # in json, keys need to be string
    layer_idx = str(layer_idx)
    if masked:
        mean, std = stats[layer_idx]["mean"], stats[layer_idx]["std"]
    else:
        mean, std = stats[layer_idx]["mean_masked"], stats[layer_idx]["std_masked"]
    return mean, std


def set_frozen_to_eval(module):
    requires_grad = []
    for p in module.parameters():
        requires_grad.append(p.requires_grad)
    if not any(requires_grad):
        module.eval()


def load_layer_stats_per_task(path, layer_idx=0):
    with open(path, "r") as f:
        stats = json.load(f)
    if layer_idx < 0:
        layer_idx = stats.keys()[layer_idx]
    # in json, keys need to be string
    layer_idx = str(layer_idx)
    # iterate tasks, build up mean and std
    means, stds = [], []
    for task in stats.keys():
        means.append(stats[task][layer_idx]["mean"])
        stds.append(stats[task][layer_idx]["std"])
    return means, stds


def make_promptcount_figures(counts, step):
    select_ratio = counts / counts.sum()
    fig1, ax1 = plt.subplots()
    ax1.bar(range(len(counts)), counts)
    ax1.set_title(f"Timestep: {str(step)}")
    ax1.set_xlabel("Prompt index")
    ax1.set_ylabel("Count")
    fig2, ax2 = plt.subplots()
    ax2.bar(range(len(select_ratio)), select_ratio)
    ax2.set_title(f"Timestep: {str(step)}")
    ax2.set_xlabel("Prompt index")
    ax2.set_ylabel("Selection ratio")
    return fig1, fig2


def make_attention_maps(attention_scores, step, lower_triu=True, vmin=None, vmax=None):
    """
    attention_scores: Tuple of `torch.FloatTensor` (one for each layer) of shape
        `(batch_size, num_heads, sequence_length,sequence_length)`.
    step: Int. Current timestep

    """
    figures = {}
    mask = None
    for i, scores in enumerate(attention_scores):
        # first attention head
        scores = scores.detach().cpu().numpy()
        h0_scores = scores[-1, 0]
        fig, ax = plt.subplots()
        if lower_triu:
            mask = np.triu(np.ones_like(h0_scores, dtype=bool))
            np.fill_diagonal(mask, False)
        sns.heatmap(h0_scores, cmap="rocket_r", mask=mask, ax=ax, vmin=vmin, vmax=vmax)
        ax.set_title(f"Timestep: {step}, Layer: {i}, Head: 0")
        figures[f"layer{i}_head0"] = fig
        # avg over all heads
        avg_scores = scores[-1].mean(0)
        fig, ax = plt.subplots()
        if lower_triu:
            mask = np.triu(np.ones_like(avg_scores, dtype=bool))
            np.fill_diagonal(mask, False)
        sns.heatmap(avg_scores, cmap="rocket_r", mask=mask, ax=ax, vmin=vmin, vmax=vmax)
        ax.set_title(f"Timestep: {step}, Layer: {i}, Head: all")
        figures[f"layer{i}_allheads"] = fig
    return figures


def make_qk_dist_plot(key, query, step):
    key, query = key.squeeze(), query.squeeze()
    df_key = pd.DataFrame(key.T, columns=[f"k{i}" for i in range(key.shape[0])])
    df_query = pd.DataFrame(query.T, columns=[f"q{i}" for i in range(query.shape[0])])
    df = pd.concat([df_key, df_query], axis=1).T
    fig, ax = plt.subplots()
    sns.heatmap(df, cmap="rocket_r", ax=ax)
    ax.set_title(f"Timestep: {str(step)}")
    ax.set_xlabel("Feature dimension")
    ax.set_ylabel("Q-K index")
    return fig


def make_sim_plot(sim, step, max_samples=5):
    """
    Make heatmap from given similarity matrix.
    Args:
        sim: np.ndarray of shape (batch_size x pool_size)
        step: Int.
        max_samples: Int. Max samples to use (across batch size). Matrix becomes unreadable for more than 10 samples.

    Returns: Matplotlib figure.

    """
    fig, ax = plt.subplots(figsize=(max_samples, sim.shape[1] * 0.3))
    if sim.shape[0] > max_samples:
        sim = sim[:max_samples]
    sns.heatmap(sim.T, cmap="rocket_r", ax=ax, annot=True)
    ax.set_title(f"Timestep: {str(step)}")
    ax.set_xlabel("Batch idx")
    ax.set_ylabel("Pool idx")
    return fig
