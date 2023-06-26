import torch
import torchvision.transforms as T
import torch.nn.functional as F


class CustomRandomCrop(T.RandomCrop):
    def __init__(self, size=84, padding=4, **kwargs):
        super().__init__(size, **kwargs)
        self.padding = padding

    def __call__(self, img):
        # first pad image by 4 pixels on each side
        img = F.pad(img, (self.padding, self.padding, self.padding, self.padding), mode='replicate')
        # crop to original size
        return super().__call__(img)

 
def make_augmentations(aug_kwargs=None):
    if aug_kwargs is None:
        aug_kwargs = {}
    aug_kwargs = aug_kwargs.copy() 
    kind = aug_kwargs.pop("kind", "crop_rotate")
    p_aug = aug_kwargs.get("p_aug", 0.5)
    if kind == "crop": 
        return T.RandomApply([CustomRandomCrop(**aug_kwargs)], p=p_aug)
    elif kind == "rotate": 
        degrees = aug_kwargs.pop("degrees", 30)
        return T.RandomApply([T.RandomRotation(degrees=degrees, **aug_kwargs)], p=p_aug)
    elif kind == "crop_rotate": 
        degrees = aug_kwargs.pop("degrees", 30)
        return T.Compose([
            T.RandomApply([CustomRandomCrop(**aug_kwargs)], p=p_aug),
            T.RandomApply([T.RandomRotation(degrees=degrees, **aug_kwargs)], p=p_aug)
        ])
    raise ValueError(f"Unknown augmentation kind: {kind}")
    
        
if __name__ == "__main__": 
    import h5py 
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    def plot_and_save(img, save_path):
        plt.imshow(img)
        plt.savefig(str(save_path))
        plt.close()

    # load single Atari trajectory
    path = "/home/thomas/Projects-Linux/CRL_with_transformers/data/atari_1M/pong/000000000.hdf5"
    with h5py.File(path, "r") as f:
        # subtrajectory only
        observations = f['states'][:]
    seq0 = torch.from_numpy(observations[0]).unsqueeze(0).float()
    seq1 = torch.from_numpy(observations[:5]).float()
    seq2 = torch.from_numpy(observations[55:60]).float()
    print(seq0.shape, seq1.shape, seq2.shape)
        
    # crop
    save_dir = Path("./figures/crop")
    crop = make_augmentations(dict(kind="crop"))
    for i, s in enumerate([seq0, seq1, seq2]): 
        augmented = crop(s)
        print(augmented.shape)
        save_dir_i = save_dir / f"seq_{i}"
        save_dir_i.mkdir(parents=True, exist_ok=True)
        for j in range(s.shape[0]):
            plot_and_save(s[j][0].numpy(), save_dir_i / f"{j}_original.png")
            plot_and_save(augmented[j][0].numpy(), save_dir_i / f"{j}_augmented.png")
            print(s[j][0].mean(), s[j][0].std())
            print(augmented[j][0].mean(), augmented[j][0].std())
                        
    # rotate
    save_dir = Path("./figures/rotation")
    rotate = make_augmentations(dict(kind="rotate"))
    for i, s in enumerate([seq0, seq1, seq2]): 
        augmented = rotate(s)
        print(augmented.shape)
        save_dir_i = save_dir / f"seq_{i}"
        save_dir_i.mkdir(parents=True, exist_ok=True)
        for j in range(s.shape[0]):
            plot_and_save(s[j][0].numpy(), save_dir_i / f"{j}_original.png")
            plot_and_save(augmented[j][0].numpy(), save_dir_i / f"{j}_augmented.png")

    # shift + rotate
    save_dir = Path("./figures/crop_rotate")
    crop_rotate = make_augmentations(dict(kind="crop_rotate"))
    for i, s in enumerate([seq0, seq1, seq2]): 
        augmented = crop_rotate(s)
        print(augmented.shape)
        save_dir_i = save_dir / f"seq_{i}"
        save_dir_i.mkdir(parents=True, exist_ok=True)
        for j in range(s.shape[0]):
            plot_and_save(s[j][0].numpy(), save_dir_i / f"{j}_original.png")
            plot_and_save(augmented[j][0].numpy(), save_dir_i / f"{j}_augmented.png")
