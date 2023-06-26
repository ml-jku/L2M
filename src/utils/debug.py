import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from datetime import datetime


class GradPlotter:

    def __init__(self, y_min=-0.01, y_max=0.5, base_dir=None):
        self.base_dir = base_dir
        if self.base_dir is None:
            self.base_dir = "/home/thomas/Projects-Linux/CRL_with_transformers/debug"
        time = datetime.now().strftime("%d-%m-%Y_%Hh%Mm")
        self.base_dir = Path(self.base_dir) / time
        self.base_dir.mkdir(exist_ok=True, parents=True)
        self.y_min = y_min
        self.y_max = y_max

    def plot_grad_flow(self, named_parameters, file_name):
        """
        Adjusted from:
            https://gist.github.com/Flova/8bed128b41a74142a661883af9e51490

        Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow

        E.g., call using:
            if self._n_updates % 1000 == 0:
                plot_grad_flow(self.critic.named_parameters(), f"critic_update,critic,step={self._n_updates}.png")
                plot_grad_flow(self.policy.named_parameters(), f"critic_update,policy,step={self._n_updates}.png")

        """
        ave_grads, max_grads, layers = [], [], []
        for n, p in named_parameters:
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().item() if p.grad is not None else 0)
                max_grads.append(p.grad.abs().max().item() if p.grad is not None else 0)
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        # plt.ylim(bottom=self.y_min, top=self.y_max)
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.savefig(self.base_dir / file_name, bbox_inches='tight')
        plt.close()
