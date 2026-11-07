"""PathMNIST seeding and one-epoch training utilities .

Call signature: train_one_epoch(model, train_loader, criterion, optimizer, device)
"""

import random

import numpy as np
import torch


def set_seed(seed=42):
    """Set Python, NumPy and PyTorch RNG seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """Initialize NumPy/Python RNGs in a DataLoader worker."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch, returning sample-weighted average loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.view(-1).long().to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    return total_loss / total_samples
