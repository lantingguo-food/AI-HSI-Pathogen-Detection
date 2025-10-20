import numpy as np

class HSITransform:
    def __init__(self, cfg):
        self.hflip = cfg["aug"].get("hflip", 0.0)
        self.vflip = cfg["aug"].get("vflip", 0.0)
        self.rot90 = cfg["aug"].get("rot90", 0.0)
        self.sj = cfg["aug"].get("spectral_jitter_std", 0.0)
        self.gn = cfg["aug"].get("gaussian_noise_std", 0.0)

    def __call__(self, patch):
        if np.random.rand() < self.hflip:
            patch = np.flip(patch, axis=1)
        if np.random.rand() < self.vflip:
            patch = np.flip(patch, axis=0)
        if np.random.rand() < self.rot90:
            k = np.random.choice([1,2,3])
            patch = np.rot90(patch, k, axes=(0,1))
        if self.sj > 0:
            patch = patch + np.random.normal(0, self.sj, size=patch.shape)
        if self.gn > 0:
            patch = patch + np.random.normal(0, self.gn, size=patch.shape)
        return patch
