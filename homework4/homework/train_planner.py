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
from homework.metrics import PlannerMetric
from homework.datasets.road_dataset import load_data


#def load_data(split_dir, batch_size=32, shuffle=False, num_workers=2):
#    dataset = RoadDataset(split_dir)
#    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def train(
    exp_dir="logs",
    model_name="mlp_planner",
    transform_pipeline="state_only",
    num_epoch=20,
    lr=1e-3,
    batch_size=32,
    seed=42,
    num_workers=4,
    **kwargs
):
    metric = PlannerMetric()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = load_model(model_name, **kwargs).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    train_loader = load_data("drive_data/train", "default", batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = load_data("drive_data/val", "default", batch_size=batch_size, num_workers=num_workers)

    best_error = float("inf")

    for epoch in range(num_epoch):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            waypoints = batch["waypoints"].to(device)
            waypoints = waypoints - waypoints[:, :1, :]

            if model_name == "cnn_planner":
                image = batch["image"].to(device)
                predictions = model(image)
            else:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                predictions = model(track_left, track_right)

            lateral_loss = criterion(predictions[..., 0], waypoints[..., 0])
            longitudinal_loss = criterion(predictions[..., 1], waypoints[..., 1])
            if model_name == "cnn_planner":
                loss = 2 * lateral_loss + 5 * longitudinal_loss
            elif model_name == "transformer_planner":
                loss = 4 * lateral_loss + longitudinal_loss
            else:
                loss = lateral_loss + longitudinal_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.add_scalar("train/loss", avg_loss, epoch)

        # Validation
        model.eval()
        metric.reset()
        with torch.no_grad():
            for batch in val_loader:
                waypoints = batch["waypoints"].to(device)
                waypoints_mask = batch["waypoints_mask"].to(device)

                if model_name == "cnn_planner":
                    image = batch["image"].to(device)
                    predictions = model(image)
                else:
                    track_left = batch["track_left"].to(device)
                    track_right = batch["track_right"].to(device)
                    predictions = model(track_left, track_right)

                metric.add(predictions, waypoints, waypoints_mask)

        val_metrics = metric.compute()
        longitudinal_error = val_metrics["longitudinal_error"]
        lateral_error = val_metrics["lateral_error"]

        logger.add_scalar("val/longitudinal_error", longitudinal_error, epoch)
        logger.add_scalar("val/lateral_error", lateral_error, epoch)

        print(f"Epoch {epoch + 1}/{num_epoch}, Loss: {avg_loss:.4f}, Lateral Error: {lateral_error:.4f}, Longitudinal Error: {longitudinal_error:.4f}")

        if longitudinal_error + lateral_error < best_error:
            best_error = longitudinal_error + lateral_error
            save_model(model)

        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default="logs")
    parser.add_argument("--model_name", default="mlp_planner")
    parser.add_argument("--transform_pipeline", default="state_only")
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    train(**vars(args))

