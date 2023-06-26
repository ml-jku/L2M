import torch.nn as nn
from stable_baselines3.common.torch_layers import FlattenExtractor, create_mlp


class FlattenExtractorWithMLP(FlattenExtractor):

    def __init__(self, observation_space, net_arch=None):
        super().__init__(observation_space)
        if net_arch is None:
            net_arch = [128, 128]

        mlp = create_mlp(self.features_dim, net_arch[-1], net_arch)
        self.mlp = nn.Sequential(*mlp)
        self._features_dim = net_arch[-1]

    def forward(self, observations):
        return self.mlp(self.flatten(observations))


def create_cwnet(
    input_dim: int,
    output_dim: int,
    net_arch=(256,256,256),
    # activation_fn=lambda: nn.LeakyReLU(negative_slope=0.2),
    activation_fn=nn.LeakyReLU,
    squash_output: bool = False,
):
    """
    Creates the same Net as described in https://arxiv.org/pdf/2105.10919.pdf
    Basically just adds LayerNorm + Tanh after first Dense layer.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0])]
    else:
        modules = []

    modules.append(nn.LayerNorm(net_arch[0]))
    modules.append(nn.Tanh())

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules
