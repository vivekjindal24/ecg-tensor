"""Training script skeleton for ECG tensor-based models.
Usage (PowerShell):
  python scripts/train.py --model cnn --epochs 5 --batch-size 16
"""
from __future__ import annotations
import argparse
from pathlib import Path
import os, sys, math, time, json, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Local imports (after adding project root to path if needed)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.ecg_preprocessing import denoise_signal, resample_signal, normalize_signal  # noqa: E402

# ------------------ Dataset Placeholder ------------------ #
class DummyEcgDataset(Dataset):
    """Synthetic dataset placeholder until real loader integrated."""
    def __init__(self, n_samples: int = 64, leads: int = 12, length: int = 1000, n_classes: int = 5):
        rng = np.random.default_rng(42)
        self.x = rng.normal(size=(n_samples, leads, length)).astype('float32')
        self.y = rng.integers(low=0, high=n_classes, size=(n_samples,)).astype('int64')
        self.n_classes = n_classes
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx])

# ------------------ Model Stubs (dup from notebook; consider refactor) ------------------ #
class ECGCnn(nn.Module):
    def __init__(self, in_channels=12, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

class ECGCnnLstm(nn.Module):
    def __init__(self, in_channels=12, num_classes=5, hidden_size=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, 7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden_size*2, num_classes)
    def forward(self, x):
        feats = self.cnn(x)  # (B,64,T)
        feats = feats.transpose(1,2)  # (B,T,64)
        out,_ = self.lstm(feats)
        out = out[:, -1]
        return self.head(out)

# ------------------ Training Utilities ------------------ #
def build_model(name: str, num_classes: int):
    if name == 'cnn':
        return ECGCnn(num_classes=num_classes)
    elif name == 'cnn_lstm':
        return ECGCnnLstm(num_classes=num_classes)
    else:
        raise ValueError(f'Unknown model: {name}')


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='cnn', choices=['cnn','cnn_lstm'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-classes', type=int, default=5)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = DummyEcgDataset(n_samples=64, n_classes=args.num_classes)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = build_model(args.model, args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs+1):
        loss = train_epoch(model, loader, criterion, optimizer, device)
        print(f'Epoch {epoch}/{args.epochs} - loss: {loss:.4f}')

    # Save checkpoint
    ckpt_dir = PROJECT_ROOT / 'artifacts' / 'models'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f'{args.model}_dummy.pth'
    torch.save({'model_state': model.state_dict(), 'config': vars(args)}, ckpt_path)
    print('Saved checkpoint to', ckpt_path)

if __name__ == '__main__':
    main()
