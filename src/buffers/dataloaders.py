from torch.utils.data import DataLoader


class MultiEpochsDataLoader(DataLoader):
    
    def __init__(self, *args, **kwargs):
        """
        From: 
            https://discuss.pytorch.org/t/enumerate-dataloader-slow/87778/4

        Ensures that the Dataset is iterated over multiple times wihthout destroying the workers. 
        
        """
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
            

if __name__ == "__main__":
    import torch
    from torch.utils.data import TensorDataset
    
    ds = TensorDataset(torch.randn(100, 10))
    loader = MultiEpochsDataLoader(ds, num_workers=4, batch_size=10)    
    loader_iter = iter(loader)
    for i in range(1000): 
        print(i)
        try: 
            b = next(loader_iter)
        except StopIteration as e: 
            loader_iter = iter(loader)
            b = next(loader_iter)
