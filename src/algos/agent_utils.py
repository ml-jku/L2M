import torch
import numpy as np
from torch.nn.parallel import DistributedDataParallel
from ..utils.loss_functions import DistanceSmoothedCrossEntropyLoss


def get_param_count(model, prefix="model"):
    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {f"{prefix}_total": params, f"{prefix}_trainable": trainable_params}


def make_loss_fn(kind, reduction="mean", label_smoothing=0.0):
    if kind in ["mse", "td3+bc"]:
        loss_fn = torch.nn.MSELoss(reduction=reduction)
    elif kind in ["smooth_l1", "dqn"]:
        loss_fn = torch.nn.SmoothL1Loss(reduction=reduction)
    elif kind == "huber":
        loss_fn = torch.nn.HuberLoss(reduction=reduction)
    elif kind == "nll":
        loss_fn = torch.nn.NLLLoss(reduction=reduction)
    elif kind == "ce":
        loss_fn = torch.nn.CrossEntropyLoss(reduction=reduction, label_smoothing=label_smoothing)
    elif kind == "dist_ce":
        loss_fn = DistanceSmoothedCrossEntropyLoss(reduction=reduction, label_smoothing=label_smoothing)
    elif kind in ["td3", "ddpg", "sac"]:
        loss_fn = None
    else:
        raise ValueError(f"Unknown loss kind: {kind}")
    return loss_fn


def make_random_proj_matrix(in_dim, proj_dim, seed=42):
    # ALWAYS first initialize the random state to deterministically get the 
    # same projection matrix (for every size)
    rng = np.random.RandomState(seed)
    return rng.normal(loc=0, scale=1.0 / np.sqrt(proj_dim), size=(proj_dim, in_dim)).astype(dtype=np.float32)


class CustomDDP(DistributedDataParallel):
    """
    The default DistributedDataParallel enforces access to class the module attributes via self.module. 
    This is impractical for our use case, as we need to access certain module access throughout. 
    We override the __getattr__ method to allow access to the module attributes directly.
    
    For example: 
    ```
        # default behaviour
        model = OnlineDecisionTransformerModel()
        model = DistributedDataParallel(model)
        model.module.some_attribute
        
        # custom behaviour using this class
        model = OnlineDecisionTransformerModel()
        model = CustomDDP(model)
        model.some_attribute
        
    ```        
    Shoudl not cause any inconsistencies: 
    https://discuss.pytorch.org/t/access-to-attributes-of-model-wrapped-in-ddp/130572
    
    """
    
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
