import os
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Inception_V3_Weights, inception_v3
import torchvision.transforms as T

from utils.torch_utils import default_device

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path: str, transform=None):
        self.files = sorted(
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        )
        if not self.files:
            raise FileNotFoundError(f"No images found in {folder_path}.")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


class FIDEvaluator:
    """Fréchet Inception Distance between a real and a generated image folder."""

    def __init__(self, device: Optional[torch.device] = None, num_workers: int = 0):
        self.device = device or default_device()
        self.num_workers = num_workers

        weights = Inception_V3_Weights.DEFAULT
        self.inception = inception_v3(weights=weights, transform_input=False, aux_logits=False)
        self.inception.fc = torch.nn.Identity()  # 2048-d pool features
        self.inception = self.inception.to(self.device).eval()

        # The weights' own preprocessing: resize/crop to 299 + matching normalization.
        self.transform = weights.transforms()
        self.transform = T.Compose([
            T.Resize(299),
            self.transform
        ])

    @torch.no_grad()
    def get_features(self, folder_path: str, batch_size: int = 32) -> np.ndarray:
        dataset = ImageFolderDataset(folder_path, transform=self.transform)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers
        )

        features = []
        for batch in dataloader:
            feat = self.inception(batch.to(self.device))
            features.append(feat.float().cpu().numpy())
        return np.concatenate(features, axis=0)

    @staticmethod
    def _statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if features.shape[0] < 2:
            raise ValueError("Need at least 2 images to estimate a covariance matrix.")
        return features.mean(axis=0), np.cov(features, rowvar=False)

    @staticmethod
    def frechet_distance(
        mu_real: np.ndarray,
        sigma_real: np.ndarray,
        mu_fake: np.ndarray,
        sigma_fake: np.ndarray,
        eps: float = 1e-6,
    ) -> float:
        diff = mu_real - mu_fake
        covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)

        if not np.isfinite(covmean).all():
            # Nudge the covariances off the singular point before retrying.
            offset = np.eye(sigma_real.shape[0]) * eps
            covmean, _ = linalg.sqrtm((sigma_real + offset).dot(sigma_fake + offset), disp=False)

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        return float(diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2.0 * covmean))

    def calculate_fid(self, real_folder: str, fake_folder: str, batch_size: int = 32) -> float:
        try:
            feat_real = self.get_features(real_folder, batch_size)
            feat_fake = self.get_features(fake_folder, batch_size)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"FID calculation failed: {e}")

        min_samples = min(feat_real.shape[0], feat_fake.shape[0])
        if min_samples < 2048:
            print(
                f"[FID warning] only {min_samples} images per set; FID is heavily biased "
                "below ~2048 samples and is not comparable to published numbers."
            )

        mu_real, sigma_real = self._statistics(feat_real)
        mu_fake, sigma_fake = self._statistics(feat_fake)
        return self.frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
