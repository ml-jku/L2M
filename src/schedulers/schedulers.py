import math


class ScheduleBase:
    def __init__(self, init_val=1.0, max_step_multiplier=1.0, max_step=None, min_val=None):
        self._max_step = max_step * max_step_multiplier
        self._min_val = min_val
        self._init_val = init_val

    @property
    def max_step(self):
        return self._max_step

    @property
    def min_val(self):
        return self._min_val

    @property
    def init_val(self):
        return self._init_val

    def __call__(self, step):
        return self.get_value(step)

    def get_value(self, step):
        raise NotImplementedError


class Linear(ScheduleBase):
    """
    Similar to :
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html#torch.optim.lr_scheduler.LinearLR
    """
    def get_value(self, step):
        val = step / self.max_step
        if self.min_val is not None and val < self.min_val:
            return self.min_val
        return val


class Step(ScheduleBase):
    def __init__(self, step_size, gamma=0.5, **kwargs):
        """
        Similar to:
            https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR
        Args:
            step_size: Int.
            gamma: Float.
        """
        super().__init__(**kwargs)
        self.step_size = step_size
        self.gamma = gamma

    def get_value(self, step):
        exponent = int(step / self.step_size)
        val = self.init_val * (self.gamma ** exponent)
        if self.min_val is not None and val < self.min_val:
            return self.min_val
        return val


class CosineAnnealing(ScheduleBase):
    def __init__(self, eta_min=0, **kwargs):
        """
        Similar to:
            https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR
        Args:
            eta_min: Float.
            **kwargs:
        """
        super().__init__(**kwargs)
        self.eta_min = eta_min

    def get_value(self, step):
        if step > self.max_step:
            return self.eta_min
        return self.eta_min + (self.init_val - self.eta_min) * (1 + math.cos(math.pi * step / self.max_step)) / 2


def make_scheduler(kind="linear", **kwargs):
    if kind == "linear":
        return Linear(**kwargs)
    elif kind == "cosine":
        return CosineAnnealing(**kwargs)
    elif kind == "step":
        return Step(**kwargs)
    return None
