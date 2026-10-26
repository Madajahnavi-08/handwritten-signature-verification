"""
PyTorch Dataset Loader for Siamese Signature Verification.
Constructs balanced pairs of Genuine-Genuine (Label 1) and Genuine-Forged (Label 0).
Author: Mada Jahnavi
Domain: Deep Learning & Computer Vision
"""

import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from preprocessing import preprocess_signature_image


class SignaturePairDataset(Dataset):
    """
    Siamese Signature Dataset.
    Generates balanced positive (label=1.0) and negative (label=0.0) image pairs.
    """
    def __init__(self, data_dir: str, num_pairs: int = 1200, is_train: bool = True, seed: int = 42):
        super(SignaturePairDataset, self).__init__()
        self.data_dir = data_dir
        self.num_pairs = num_pairs
        self.is_train = is_train
        self.rng = random.Random(seed)

        # Locate raw files
        self.genuine_dir = os.path.join(data_dir, "raw", "genuine")
        self.forged_dir = os.path.join(data_dir, "raw", "forged")

        # Map signers to their files
        self.user_genuine_map = {}
        self.user_forged_map = {}

        gen_files = glob.glob(os.path.join(self.genuine_dir, "*.png"))
        for f in gen_files:
            basename = os.path.basename(f)
            u_id = basename.split("_genuine_")[0]
            if u_id not in self.user_genuine_map:
                self.user_genuine_map[u_id] = []
            self.user_genuine_map[u_id].append(f)

        forg_files = glob.glob(os.path.join(self.forged_dir, "*.png"))
        for f in forg_files:
            basename = os.path.basename(f)
            u_id = basename.split("_forged_")[0]
            if u_id not in self.user_forged_map:
                self.user_forged_map[u_id] = []
            self.user_forged_map[u_id].append(f)

        self.users = sorted(list(self.user_genuine_map.keys()))
        if len(self.users) == 0:
            raise ValueError(f"No signature files found in {self.genuine_dir}. Run generate_dataset.py first!")

        # Augmentation transforms for training pairs
        self.train_transforms = transforms.Compose([
            transforms.RandomRotation(degrees=(-6, 6)),
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.95, 1.05)),
        ])

        # Pre-build fixed pairs list for deterministic evaluation & epoch consistency
        self.pairs = self._build_pairs()

    def _build_pairs(self):
        pairs = []
        half = self.num_pairs // 2

        # 1. Positive Pairs (Genuine A from User i, Genuine B from User i) -> Label = 1.0
        for _ in range(half):
            u_id = self.rng.choice(self.users)
            user_gens = self.user_genuine_map[u_id]
            if len(user_gens) >= 2:
                img1_path, img2_path = self.rng.sample(user_gens, 2)
            else:
                img1_path = img2_path = user_gens[0]
            pairs.append((img1_path, img2_path, 1.0))

        # 2. Negative Pairs (Label = 0.0)
        # Mix of skilled forgeries (same user name, forged file) and random forgeries (different user)
        for i in range(half):
            u_id1 = self.rng.choice(self.users)
            img1_path = self.rng.choice(self.user_genuine_map[u_id1])

            if i % 2 == 0 and u_id1 in self.user_forged_map and len(self.user_forged_map[u_id1]) > 0:
                # Skilled forgery
                img2_path = self.rng.choice(self.user_forged_map[u_id1])
            else:
                # Random forgery from another user
                other_users = [u for u in self.users if u != u_id1]
                u_id2 = self.rng.choice(other_users)
                img2_path = self.rng.choice(self.user_genuine_map[u_id2])

            pairs.append((img1_path, img2_path, 0.0))

        self.rng.shuffle(pairs)
        return pairs

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, path: str) -> torch.Tensor:
        # Preprocess through the structured pipeline
        proc_arr = preprocess_signature_image(path)
        tensor = torch.from_numpy(proc_arr).unsqueeze(0)  # (1, 100, 100)

        # Apply mild data augmentation only during training
        if self.is_train:
            tensor = self.train_transforms(tensor)
        return tensor

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        t1 = self._load_image(img1_path)
        t2 = self._load_image(img2_path)
        y = torch.tensor([label], dtype=torch.float32)
        return t1, t2, y


def get_dataloaders(data_dir: str, batch_size: int = 32, num_train_pairs: int = 1400, num_val_pairs: int = 300):
    """
    Constructs PyTorch DataLoader instances for training and validation.
    """
    train_dataset = SignaturePairDataset(data_dir, num_pairs=num_train_pairs, is_train=True, seed=42)
    val_dataset = SignaturePairDataset(data_dir, num_pairs=num_val_pairs, is_train=False, seed=99)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
