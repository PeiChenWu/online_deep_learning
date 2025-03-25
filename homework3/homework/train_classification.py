import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
from torch.utils.data import DataLoader

from homework.models import ClassificationLoss, load_model, save_model
from homework.datasets.classification_dataset import SuperTuxDataset


def load_data(split_dir, transform_pipeline="default", batch_size=128, shuffle=False, num_workers=2):
    dataset = SuperTuxDataset(split_dir, transform_pipeline)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def train(exp_dir="logs", model_name="classifier", num_epoch=50, lr=1e-3, batch_size=128, seed=42, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = load_model(model_name, **kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = ClassificationLoss()

    train_loader = load_data("classification_data/train", "aug", batch_size, shuffle=True)
    val_loader = load_data("classification_data/val", "default", batch_size)

    best_acc = 0.0
    global_step = 0

    for epoch in range(num_epoch):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in train_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)       

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (torch.argmax(logits, dim=1) == y).sum().item()
            total += y.size(0)

            logger.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        train_acc = correct / total
        val_acc = evaluate(model, val_loader, device)
        logger.add_scalar("train/acc", train_acc, epoch)
        logger.add_scalar("val/acc", val_acc, epoch)

        print(f"[Epoch {epoch}] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_model(log_dir, model)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.inference_mode():
        for batch in loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct += (torch.argmax(logits, dim=1) == y).sum().item()
            total += y.size(0)
    return correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default="logs")
    parser.add_argument("--model_name", default="classifier")
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(**vars(args))
