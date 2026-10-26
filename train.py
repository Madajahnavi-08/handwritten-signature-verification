"""
Training Pipeline for Siamese Neural Network Signature Verification.
Author: Mada Jahnavi
Domain: Deep Learning & Computer Vision
"""

import os
import json
import time
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

from model import SiameseSignatureNet, ContrastiveLoss
from dataset import get_dataloaders


def train_siamese_model(
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    margin: float = 1.0,
    save_path: str = None
):
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_dir, "data")
    models_dir = os.path.join(project_dir, "saved_models")
    os.makedirs(models_dir, exist_ok=True)
    
    if save_path is None:
        save_path = os.path.join(models_dir, "best_siamese_model.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    # Build Dataloaders
    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_train_pairs=1200,
        num_val_pairs=300
    )

    # Initialize Model, Loss Function, and Optimizer
    model = SiameseSignatureNet(embedding_dim=128).to(device)
    criterion = ContrastiveLoss(margin=margin)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_distances_genuine": [],
        "val_distances_forged": []
    }

    best_val_loss = float("inf")
    best_accuracy = 0.0
    start_time = time.time()

    print("==================================================================")
    print("      Starting Siamese Neural Network Training Pipeline")
    print(f"      Epochs: {epochs} | Batch Size: {batch_size} | Margin: {margin}")
    print("==================================================================")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # ----------------- Training Step -----------------
        model.train()
        running_train_loss = 0.0
        
        for img1, img2, labels in train_loader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * img1.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        # ----------------- Validation Step -----------------
        model.eval()
        running_val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        gen_distances = []
        forg_distances = []

        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1 = img1.to(device)
                img2 = img2.to(device)
                labels = labels.to(device)

                out1, out2 = model(img1, img2)
                loss = criterion(out1, out2, labels)
                running_val_loss += loss.item() * img1.size(0)

                # Compute Euclidean distances
                distances = F.pairwise_distance(out1, out2).cpu().numpy()
                labels_np = labels.squeeze().cpu().numpy()

                for d, lbl in zip(distances, labels_np):
                    if lbl == 1.0:
                        gen_distances.append(float(d))
                        # Genuine pair: predicted genuine if distance <= threshold (0.5 * margin)
                        if d <= (margin * 0.5):
                            correct_predictions += 1
                    else:
                        forg_distances.append(float(d))
                        # Forged pair: predicted forged if distance > (0.5 * margin)
                        if d > (margin * 0.5):
                            correct_predictions += 1
                    total_predictions += 1

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = (correct_predictions / total_predictions) * 100.0
        scheduler.step(epoch_val_loss)

        # Record metrics
        history["train_loss"].append(round(epoch_train_loss, 5))
        history["val_loss"].append(round(epoch_val_loss, 5))
        history["val_accuracy"].append(round(epoch_val_acc, 2))
        avg_gen_d = float(np.mean(gen_distances)) if gen_distances else 0.0
        avg_forg_d = float(np.mean(forg_distances)) if forg_distances else 0.0
        history["val_distances_genuine"].append(round(avg_gen_d, 4))
        history["val_distances_forged"].append(round(avg_forg_d, 4))

        epoch_duration = time.time() - epoch_start
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_duration:.1f}s) - "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {epoch_val_acc:.2f}% | "
              f"Avg Dist [Gen: {avg_gen_d:.3f}, Forg: {avg_forg_d:.3f}]")

        # Save checkpoint if best validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_accuracy = epoch_val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_accuracy": best_accuracy,
                "margin": margin
            }, save_path)
            print(f"  --> Checkpoint saved: New lowest val loss: {best_val_loss:.4f} (Acc: {best_accuracy:.2f}%)")

    total_training_time = time.time() - start_time
    print("==================================================================")
    print(f"Training Complete in {total_training_time:.1f}s!")
    print(f"Best Validation Accuracy: {best_accuracy:.2f}% | Best Loss: {best_val_loss:.4f}")
    print(f"Saved Model Path: {save_path}")
    print("==================================================================")

    # Save metrics history to JSON
    metrics_path = os.path.join(models_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=4)

    # Plot Loss & Distance Curves
    plot_path = os.path.join(models_dir, "training_curves.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Curve
    ax1.plot(range(1, epochs + 1), history["train_loss"], marker='o', label='Train Contrastive Loss', color='#2563eb')
    ax1.plot(range(1, epochs + 1), history["val_loss"], marker='s', label='Val Contrastive Loss', color='#dc2626')
    ax1.set_title('Siamese Network Loss Progression', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Contrastive Loss')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # Feature Separation Distance Curve
    ax2.plot(range(1, epochs + 1), history["val_distances_genuine"], marker='^', label='Genuine Pair Dist (Target -> 0)', color='#16a34a')
    ax2.plot(range(1, epochs + 1), history["val_distances_forged"], marker='v', label='Forged Pair Dist (Target >= 1.0)', color='#9333ea')
    ax2.axhline(y=0.5, color='gray', linestyle=':', label='Decision Threshold (0.5)')
    ax2.set_title('Feature Distance Separation (Genuine vs Forged)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Euclidean Distance')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Training curves saved to: {plot_path}")

    return history


if __name__ == "__main__":
    train_siamese_model(epochs=12, batch_size=32, learning_rate=0.001)
