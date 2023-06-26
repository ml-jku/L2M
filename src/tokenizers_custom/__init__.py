from .base_tokenizer import BaseTokenizer
from .mu_law_tokenizer import MuLawTokenizer
from .minmax_tokenizer import MinMaxTokenizer, MinMaxTokenizer2


def make_tokenizer(kind, tokenizer_kwargs=None):
    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}
    if kind == 'mulaw':
        return MuLawTokenizer(**tokenizer_kwargs)
    elif kind == 'minmax':
        return MinMaxTokenizer(**tokenizer_kwargs)
    elif kind == 'minmax2':
        return MinMaxTokenizer2(**tokenizer_kwargs)
    else:
        raise ValueError(f"Unknown tokenizer type {kind}")
