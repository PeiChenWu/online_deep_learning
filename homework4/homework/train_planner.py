"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

"""
Usage:
    python3 -m homework.train_planner --your_args here
"""


import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
from torch.utils.data import DataLoader

from homework.models import load_model, save_model
from homework.datasets.road_dataset import RoadDataset
from homework.metrics import calculate_longitudinal_error, calculate_lateral_error


def load_data(split_dir, batch_size=32, shuffle=False, num_workers=2):
    dataset = RoadDataset(split_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def train(exp_dir="logs", model_name="mlp_planner", num_epoch=20, lr=1e-3, batch_size=32, seed=42, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = load_model(model_name, **kwargs).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = load_data("drive_data/train", batch_size=batch_size, shuffle=True)
    val_loader = load_data("drive_data/val", batch_size=batch_size)

    best_error = float("inf")

    for epoch in range(num_epoch):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            track_left = batch["track_left"].to(device)
            track_right = batch["track_right"].to(device)
            waypoints = batch["waypoints"].to(device)

            optimizer.zero_grad()
            predictions = model(track_left, track_right)
            loss = criterion(predictions, waypoints)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.add_scalar("train/loss", avg_loss, epoch)

        # Validation
        model.eval()
        total_lateral_error, total_longitudinal_error = 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                waypoints = batch["waypoints"].to(device)
                predictions = model(track_left, track_right)

                lateral_error = calculate_lateral_error(predictions, waypoints)
                longitudinal_error = calculate_longitudinal_error(predictions, waypoints)

                total_lateral_error += lateral_error.item()
                total_longitudinal_error += longitudinal_error.item()

        avg_lateral_error = total_lateral_error / len(val_loader)
        avg_longitudinal_error = total_longitudinal_error / len(val_loader)

        logger.add_scalar("val/lateral_error", avg_lateral_error, epoch)
        logger.add_scalar("val/longitudinal_error", avg_longitudinal_error, epoch)

        print(f"Epoch {epoch + 1}/{num_epoch}, Loss: {avg_loss:.4f}, Lateral Error: {avg_lateral_error:.4f}, Longitudinal Error: {avg_longitudinal_error:.4f}")

        if avg_lateral_error + avg_longitudinal_error < best_error:
            best_error = avg_lateral_error + avg_longitudinal_error
            save_model(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default="logs")
    parser.add_argument("--model_name", default="mlp_planner")
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(**vars(args))

