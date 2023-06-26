"""
Adjusted from: https://github.com/G-Wang/WaveRNN-Pytorch/blob/master/utils.py

Could also just use torchaudio:
 https://github.com/pytorch/audio/blob/0cd25093626d067e008e1f81ad76e072bd4a1edd/torchaudio/transforms.py#L757

"""
import torch
import numpy as np
from .base_tokenizer import BaseTokenizer


class MuLawTokenizer(BaseTokenizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def tokenize(self, x):
        """
        Encode signal based on mu-law companding.  For more info see the
        `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_
        This algorithm assumes the signal has been scaled to between -1 and 1 and
        returns a signal encoded with values from 0 to quantization_channels - 1
        Args:
            quantization_channels (int): Number of channels. default: 256
        """
        mu = self.vocab_size - 1
        if isinstance(x, np.ndarray):
            x_mu = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
            tokens = ((x_mu + 1) / 2 * mu + 0.5).astype(int)
            if self.shift != 0:
                return tokens + self.shift
            return tokens
        elif isinstance(x, (torch.Tensor, torch.LongTensor)):
            if isinstance(x, torch.LongTensor):
                x = x.float()
            mu = torch.FloatTensor([mu]).to(device=x.device)
            x_mu = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(mu)
            tokens = ((x_mu + 1) / 2 * mu + 0.5).long()
            if self.shift != 0:
                return tokens + self.shift
            return tokens
        raise NotImplementedError()

    def inv_tokenize(self, x_mu):
        """
        Decode mu-law encoded signal.  For more info see the
        `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_
        This expects an input with values between 0 and quantization_channels - 1
        and returns a signal scaled between -1 and 1.
        Args:
            quantization_channels (int): Number of channels. default: 256
        """
        mu = self.vocab_size - 1.
        if self.shift != 0:
            x_mu = x_mu - self.shift
        if isinstance(x_mu, np.ndarray):
            x = ((x_mu) / mu) * 2 - 1.
            return np.sign(x) * (np.exp(np.abs(x) * np.log1p(mu)) - 1.) / mu
        elif isinstance(x_mu, (torch.Tensor, torch.LongTensor)):
            if isinstance(x_mu, (torch.LongTensor, torch.cuda.LongTensor)):
                x_mu = x_mu.float()
            mu = torch.FloatTensor([mu]).to(x_mu.device)
            x = ((x_mu) / mu) * 2 - 1.
            return torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(mu)) - 1.) / mu
        raise NotImplementedError()
