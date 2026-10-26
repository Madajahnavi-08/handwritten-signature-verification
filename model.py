"""
Siamese Neural Network Architecture & Contrastive Loss for Offline Handwritten Signature Verification.
Author: Mada Jahnavi
Domain: Deep Learning & Computer Vision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SignatureCNNBranch(nn.Module):
    """
    Feature Extractor CNN Twin Branch.
    Extracts a 128-dimensional discriminative embedding from a preprocessed grayscale signature image.
    """
    def __init__(self, embedding_dim: int = 128):
        super(SignatureCNNBranch, self).__init__()
        
        # Block 1: Input (1 x 100 x 100) -> Output (32 x 50 x 50)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2: Input (32 x 50 x 50) -> Output (64 x 25 x 25)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3: Input (64 x 25 x 25) -> Output (128 x 12 x 12)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 4: Input (128 x 12 x 12) -> Output (256 x 6 x 6)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Fully Connected Embedding Head
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.35),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.flatten(x, 1)
        embedding = self.fc(x)
        # L2 Normalization projects embeddings onto a unit hypersphere (range [0, 2] for Euclidean distance)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


class SiameseSignatureNet(nn.Module):
    """
    Siamese Architecture for Signature Pair Comparison.
    Uses weight-sharing twin branches to project two signature images into feature space.
    """
    def __init__(self, embedding_dim: int = 128):
        super(SiameseSignatureNet, self).__init__()
        self.branch = SignatureCNNBranch(embedding_dim=embedding_dim)

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embedding from a single signature image."""
        return self.branch(x)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        """Forward pass through both twins sharing the identical weight parameters."""
        out1 = self.forward_once(img1)
        out2 = self.forward_once(img2)
        return out1, out2


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss Function (Hadsell et al. 2006).
    
    Formula:
      L = (1/2) * [ y * D^2 + (1 - y) * max(0, margin - D)^2 ]
    
    Where:
      - y = 1 for GENUINE pairs (similar samples, forces D -> 0)
      - y = 0 for FORGED / DISSIMILAR pairs (forces D >= margin)
      - D = ||f(x1) - f(x2)||_2 (Euclidean Distance)
      - margin = separation threshold (default: 1.0)
    """
    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, out1: torch.Tensor, out2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # Euclidean distance between normalized feature vectors
        euclidean_distance = F.pairwise_distance(out1, out2, keepdim=True)
        
        # Loss calculation
        loss_genuine = label * torch.pow(euclidean_distance, 2)
        loss_forged = (1.0 - label) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2
        )
        
        loss = 0.5 * torch.mean(loss_genuine + loss_forged)
        return loss


def compute_similarity(distance: float, threshold: float = 0.18, margin: float = 0.50) -> dict:
    """
    Converts Euclidean Distance to Human-Understandable Metrics for Banking Decision.
    
    Returns:
      - distance: raw Euclidean distance [0, 2]
      - similarity_pct: percentage similarity [0, 100]
      - is_genuine: bool verification verdict
      - confidence: confidence score [0, 100]
      - verdict_text: user-facing banking decision
    """
    distance = max(0.0, float(distance))
    
    # Sigmoidal / exponential similarity scaling relative to decision threshold
    # At d=0 -> 99.8%, at d=threshold -> ~75%, at d=2*threshold -> ~35%, at d>3*threshold -> <15%
    ratio = distance / (threshold + 1e-6)
    if ratio <= 1.0:
        # Distance within genuine boundary [0, threshold]
        similarity_pct = 99.5 - (ratio * 25.0)  # 99.5% down to 74.5%
        confidence = 99.0 - (ratio * 30.0)
        is_genuine = True
        verdict_text = "GENUINE SIGNATURE VERIFIED"
    else:
        # Distance beyond threshold (forgery / mismatch)
        excess = ratio - 1.0
        similarity_pct = max(5.0, 70.0 - (excess * 35.0))
        confidence = min(99.0, 65.0 + (excess * 25.0))
        is_genuine = False
        verdict_text = "SUSPECTED FORGERY / MISMATCH"

    return {
        "distance": round(distance, 4),
        "similarity_pct": round(similarity_pct, 1),
        "is_genuine": is_genuine,
        "confidence": round(confidence, 1),
        "verdict_text": verdict_text
    }

    return {
        "distance": round(distance, 4),
        "similarity_pct": round(similarity_pct, 2),
        "is_genuine": is_genuine,
        "confidence": round(confidence, 2),
        "verdict_text": verdict_text
    }
