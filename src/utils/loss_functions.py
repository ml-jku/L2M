import torch.nn as nn
import torch
import torch.nn.functional as F


class LogitNormLoss(nn.Module):

    def __init__(self, t=1.0, reduction="mean"):
        """
        # From: https://github.com/hongxin001/logitnorm_ood/blob/main/common/loss_function.py

        """
        super(LogitNormLoss, self).__init__()
        self.t = t
        self.reduction = reduction

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        return F.cross_entropy(logit_norm, target, reduction=self.reduction)


class DistanceSmoothedCrossEntropyLoss(torch.nn.Module):
    
    def __init__(self, label_smoothing=0.1, ignore_index=-100, reduction='mean'):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: the model's output logits
        targets: the true class labels
        """
        num_classes = logits.size(-1)

        # calculate smooth targets via distance from true class
        distances = torch.abs(torch.arange(num_classes, device=targets.device) - targets.unsqueeze(1))
        inv_distances = 1 / distances
        inv_distances[inv_distances == float('inf')] = 0
        inv_distances_norm = inv_distances / inv_distances.sum(dim=-1, keepdim=True)
        smooth_targets = inv_distances_norm * self.label_smoothing
        # set the weights for the true class
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.label_smoothing)
        loss = F.cross_entropy(logits, smooth_targets, reduction='none')

        # mask out the ignore index if present
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            loss = loss[mask]

        # reduce the loss if needed
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
