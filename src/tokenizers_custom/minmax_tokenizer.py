import torch
from .base_tokenizer import BaseTokenizer


class MinMaxTokenizer(BaseTokenizer):

    def __init__(self, min_val=-1, max_val=1, **kwargs):
        super().__init__(**kwargs)
        self.min_val = min_val
        self.max_val = max_val
        self.bin_width = (max_val - min_val) / self.vocab_size

    def tokenize(self, x):
        # Reshape the input tensor to have shape (batch_size, num_features)
        batch_size, num_features = x.shape[0], x.shape[1:]
        x = x.view(batch_size, -1)

        # Compute the indices of the bins
        tokens = ((x - self.min_val) / self.bin_width).long().clamp(min=0, max=self.vocab_size - 1)

        # Reshape the output tensor to have the same shape as the input tensor
        tokens = tokens.view(batch_size, *num_features)
        
        if self.shift != 0:
            return tokens + self.shift
        return tokens

    def inv_tokenize(self, x):
        if self.shift != 0: 
            x = x - self.shift
            # can't be smaller than 0 
            x[x < 0] = 0
                
        # Reshape the input tensor to have shape (batch_size, num_features)
        batch_size, num_features = x.shape[0], x.shape[1:]
        x = x.view(batch_size, -1)

        # Compute the values of the bins
        values = x.float() * self.bin_width + self.min_val

        # Reshape the output tensor to have the same shape as the input tensor
        return values.view(batch_size, *num_features)


class MinMaxTokenizer2(BaseTokenizer):

    def __init__(self, min_val=-1, max_val=1, **kwargs):
        """
        Tokenizes a given (action) input as described by: https://arxiv.org/abs/2212.06817
        Args:
            **kwargs:
            
        """
        super().__init__(**kwargs)
        self.min_val = min_val
        self.max_val = max_val

    def tokenize(self, x):
        x = torch.clamp(x, self.min_val, self.max_val)
        # Normalize the action [batch, actions_size]
        tokens = (x - self.min_val) / (self.max_val - self.min_val)
        # Bucket and discretize the action to vocab_size, [batch, actions_size]
        tokens = (tokens * (self.vocab_size - 1)).long()
        if self.shift != 0:
            return tokens + self.shift
        return tokens

    def inv_tokenize(self, x):
        if self.shift != 0: 
            x = x - self.shift
            x[x < 0] = 0
        x = x.float() / (self.vocab_size - 1)
        x = (x * (self.max_val - self.min_val)) + self.min_val
        return x
