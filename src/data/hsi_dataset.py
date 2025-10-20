import os
import numpy as np
import torch
from torch.utils.data import Dataset
from .transforms import HSITransform
from ..utils.hsi_ops import NORMALIZERS
try:
    import spectral as sp
except Exception:
    sp = None

class HSICubeDataset(Dataset):
    def __init__(self, root_dir, cfg, split="train"):
        self.root = os.path.join(root_dir)
        self.cfg = cfg
        self.split = split
        self.winH, self.winW = cfg["data"]["window"]
        self.strideH, self.strideW = cfg["data"]["stride"]
        self.input_format = cfg["data"]["input_format"]
        self.normalize = cfg["data"]["normalize"]
        self.spec_bands = cfg["data"].get("spec_bands", None)
        self.aug = HSITransform(cfg) if split == "train" else None
        labels_csv = os.path.join(root_dir, cfg["data"]["labels_file"])
        with open(labels_csv, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        self.items = []
        for line in lines:
            name, lab = line.split(",")
            self.items.append((os.path.join(root_dir, cfg["data"]["cube_key"], name), int(lab)))
        self._index_patches()

    def _load_cube(self, path):
        if self.input_format == "npy":
            X = np.load(path)  # H x W x C
        elif self.input_format == "envi":
            assert sp is not None, "Install 'spectral' for ENVI support."
            img = sp.open_image(path.replace(".img", ".hdr"))
            X = np.array(img.load())
        else:
            raise ValueError("Unsupported input_format")
        if self.spec_bands is not None:
            X = X[..., self.spec_bands]
        X = NORMALIZERS[self.normalize](X)
        return X.astype(np.float32)

    def _index_patches(self):
        self.index = []
        self.cubes = []
        for i, (p, _) in enumerate(self.items):
            X = self._load_cube(p)
            H, W, C = X.shape
            for y in range(0, max(1, H - self.winH + 1), self.strideH):
                for x in range(0, max(1, W - self.winW + 1), self.strideW):
                    self.index.append((i, y, x))
            self.cubes.append(X)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        cube_idx, y, x = self.index[idx]
        path, label = self.items[cube_idx]
        X = self.cubes[cube_idx]
        patch = X[y:y+self.winH, x:x+self.winW, :]
        if patch.shape[0] < self.winH or patch.shape[1] < self.winW:
            padH = self.winH - patch.shape[0]
            padW = self.winW - patch.shape[1]
            patch = np.pad(patch, ((0,padH),(0,padW),(0,0)), mode='reflect')
        if self.aug is not None:
            patch = self.aug(patch)
        patch = np.transpose(patch, (2, 0, 1))  # C x H x W
        patch = np.expand_dims(patch, axis=0)   # 1 x C x H x W  (depth=C)
        return {
            "x": torch.from_numpy(patch),
            "y": torch.tensor(label, dtype=torch.long),
            "meta": {"cube_path": path, "y": y, "x": x}
        }
