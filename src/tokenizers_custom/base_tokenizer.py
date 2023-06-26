class BaseTokenizer:

    def __init__(self, vocab_size=256, shift=0):
        self._vocab_size = vocab_size
        self._shift = shift

    def tokenize(self, x):
        raise NotImplementedError()

    def inv_tokenize(self, x):
        raise NotImplementedError()

    @property
    def vocab_size(self):
        return self._vocab_size
    
    @property
    def shift(self):
        return self._shift
