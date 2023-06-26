import gym
import torch
import torch.nn as nn
import torchvision
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.preprocessing import is_image_space
from .adapter import Adapter


class ImpalaCNN(BaseFeaturesExtractor):
    """
    CNN from IMPLALA paper:
        - https://arxiv.org/abs/1802.01561
        
    Implementation adapted from: 
        - https://github.com/schmidtdominik/Rainbow/blob/main/common/networks.py

    """ 
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, model_size=1, 
                 spectral_norm=False, use_adapters=False):
        super().__init__(observation_space, features_dim)        
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False), (
            "You should use ImpalaCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )
        def identity(p): return p
        self.use_adapters = use_adapters
        norm_func = torch.nn.utils.spectral_norm if (spectral_norm == 'all') else identity
        norm_func_last = torch.nn.utils.spectral_norm if (spectral_norm == 'last' or spectral_norm == 'all') else identity
        n_input_channels = observation_space.shape[0]
        
        # modullelist to allow for modulation in forward()
        self.cnn = nn.ModuleList([
            ImpalaCNNBlock(n_input_channels, 16 * model_size, norm_func=norm_func),
            ImpalaCNNBlock(16 * model_size, 32 * model_size, norm_func=norm_func),
            ImpalaCNNBlock(32 * model_size, 32 * model_size, norm_func=norm_func_last)
        ])
        self.act_flatten = nn.Sequential(nn.ReLU(), nn.Flatten()) 

        # Compute shape by doing one forward pass
        with torch.no_grad():
            dummy = torch.as_tensor(observation_space.sample()[None]).float()
            for block in self.cnn:
                dummy = block(dummy)
            n_flatten = self.act_flatten(dummy).shape[1]
            del dummy
        
        if self.use_adapters: 
            self.adapter1 = Adapter(n_flatten, n_flatten )
            self.adapter2 = Adapter(features_dim, features_dim)

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        self.convs_per_block = 5
        
    def forward(self, x, mod_vectors=None):
        for i, block in enumerate(self.cnn):
            layer_idx = i * self.convs_per_block
            mods = mod_vectors[layer_idx: layer_idx + self.convs_per_block] if mod_vectors and len(mod_vectors) > 1 else None
            x = block(x, mod_vectors=mods)
        x = self.act_flatten(x)
        if self.use_adapters:
            x, _, _ = self.adapter1(x, x)
        x = self.linear(x)        
        if mod_vectors: 
            x = x * mod_vectors[-1]
        if self.use_adapters:
            x, _, _ = self.adapter2(x, x)
        return x
    
    def get_layer_out_dims(self):
        out_dims = []
        for block in self.cnn: 
            out_dims += block.get_out_channels()
        out_dims.append(self.features_dim)
        return out_dims      
    

class ImpalaCNNResidual(nn.Module):
    """
    Simple residual block used in the large IMPALA CNN.
    
    Implementation adapted from: 
        - https://github.com/schmidtdominik/Rainbow/blob/main/common/networks.py
    
    """
    def __init__(self, depth, norm_func):
        super().__init__()
        self.depth = depth
        self.relu = nn.ReLU()
        self.conv_0 = norm_func(nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1))
        self.conv_1 = norm_func(nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1))

    def forward(self, x, mod_vectors=None):
        x_ = self.conv_0(self.relu(x))
        if mod_vectors:
            x_ = x_ * mod_vectors[0]
        x_ = self.conv_1(self.relu(x_))
        if mod_vectors: 
            x_ = x_ * mod_vectors[1]            
        return x + x_
    
    def get_out_channels(self): 
        return [self.depth] * 2


class ImpalaCNNBlock(nn.Module):
    """
    Three of these blocks are used in the large IMPALA CNN.
    Implementation adapted from: 
        - https://github.com/schmidtdominik/Rainbow/blob/main/common/networks.py
    
    """
    def __init__(self, depth_in, depth_out, norm_func):
        super().__init__()
        self.depth_in = depth_in
        self.depth_out = depth_out
        self.conv = nn.Conv2d(in_channels=depth_in, out_channels=depth_out, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(3, 2, padding=1)
        self.residual_0 = ImpalaCNNResidual(depth_out, norm_func=norm_func)
        self.residual_1 = ImpalaCNNResidual(depth_out, norm_func=norm_func)

    def forward(self, x, mod_vectors=None):
        x = self.conv(x)
        if mod_vectors: 
            x = x * mod_vectors[0]
        x = self.max_pool(x)
        x = self.residual_0(x, mod_vectors=mod_vectors[1:3] if mod_vectors else None)
        x = self.residual_1(x, mod_vectors=mod_vectors[3:5] if mod_vectors else None)
        return x
    
    def get_out_channels(self):
        return [self.depth_out, *self.residual_0.get_out_channels(), *self.residual_1.get_out_channels()]
        

class EfficientNetSb3(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, frozen=True, 
                 unfreeze_last_block=True):
        super().__init__(observation_space, features_dim)        
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False), (
            "You should use ImpalaCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )
        self.frozen = frozen
        self.unfreeze_last_block = unfreeze_last_block
        # make efficient net in torchvision
        self.cnn = torchvision.models.efficientnet_b3(pretrained=True)
        
        # we don't need the classifier head, but want to have the output features
        self.cnn.classifier = nn.Identity()
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample = observation_space.sample()[None]
            if not sample.shape[1] == 3:
                sample = np.repeat(sample, 3, axis=1)
            n_flatten = self.cnn(torch.as_tensor(sample).float()).shape[1]
            del sample

        # additional learnable linear projection
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

        if self.frozen: 
            # Freeze all original parameters - only Linear projection should be trainable
            for p in self.cnn.parameters():
                p.requires_grad_(False)
            # disable batchnorm running stat update
            self.disable_batchnorm()
        if self.unfreeze_last_block: 
            for name, p in self.cnn.named_parameters():
                if "features.8" in name or "features.7" in name:
                    p.requires_grad_(True)

    def forward(self, observations, mod_vectors=None):        
        return self.linear(self.cnn(observations)) 
    
    def train(self, mode=True):
        super().train(mode)
        if self.frozen: 
            self.disable_batchnorm()
    
    def disable_batchnorm(self):
        for module in self.cnn.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                
    
def make_image_encoder(observation_space, features_dim, encoder_kwargs=None):
    if encoder_kwargs is None:
        encoder_kwargs = {}
    encoder_kwargs = encoder_kwargs.copy()
    kind = encoder_kwargs.pop("kind", "impala")
    if kind == "nature":
        return NatureCNN(observation_space=observation_space, features_dim=features_dim, **encoder_kwargs)
    elif kind == "impala":
        return ImpalaCNN(observation_space=observation_space, features_dim=features_dim, **encoder_kwargs)
    elif kind == "efficientnet": 
        return EfficientNetSb3(observation_space=observation_space, features_dim=features_dim, **encoder_kwargs)
    else:
        raise ValueError(f"Unknown image encoder: {kind}")
