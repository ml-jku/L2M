import torch
from .sam import SAM
from .lamb import Lamb
from .adan import Adan
from .lion import Lion
from .sophia import Sophia, SophiaG


def filter_params(params):
    # filter out params that don't require gradients
    params = list(params)
    if isinstance(params[0], dict):
        params_filtered = []
        for group in params:
            group['params'] = filter(lambda p: p.requires_grad, group['params'])
            params_filtered.append(group)
        params = params_filtered
    else:
        params = filter(lambda p: p.requires_grad, params)
    return params


def make_optimizer(kind, params, lr):
    params = filter_params(params)
    if kind == "adamw":
        optimizer = torch.optim.AdamW(params, lr=lr)
    elif kind == "adam":
        optimizer = torch.optim.Adam(params, lr=lr)
    elif kind == "radam":
        optimizer = torch.optim.RAdam(params, lr=lr)
    elif kind == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr)
    elif kind == "rmsprop":
        optimizer = torch.optim.RMSprop(params, lr=lr)
    elif kind == "sam":
        base_optimizer = torch.optim.SGD
        optimizer = SAM(params, base_optimizer, lr=lr, momentum=0.9)
    elif kind == "lamb":
        optimizer = Lamb(params, lr=lr)
    elif kind == "adan":
        optimizer = Adan(params, lr=lr)
    elif kind == "lion":
        optimizer = Lion(params, lr=lr)
    elif kind == "sophia":
        optimizer = Sophia(params, lr=lr)
    elif kind == "sophia_g":
        optimizer = SophiaG(params, lr=lr)
    else:
        raise NotImplementedError()
    return optimizer
