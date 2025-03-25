import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
from torch.utils.data import DataLoader

from homework.models import Detector, save_model
from homework.metrics import ConfusionMatrix
from homework.datasets.road_dataset import RoadDataset


def load_data(split_dir, transform_pipeline="default", batch_size=32, shuffle=False, num_workers=2):
    dataset = RoadDataset(split_dir, transform_pipeline)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def train(exp_dir="logs", model_name="detector", num_epoch=30, lr=1e-3, batch_size=32, seed=42, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = Detector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    seg_loss_fn = torch.nn.CrossEntropyLoss()
    depth_loss_fn = torch.nn.L1Loss()


    train_loader = load_data("drive_data/train", "aug", batch_size, shuffle=True)
    val_loader = load_data("drive_data/val", "default", batch_size)

    best_miou = 0.0

    for epoch in range(num_epoch):
        model.train()
        seg_cm = ConfusionMatrix(num_classes=3)
        total_depth_loss = 0.0

        for batch in train_loader:
            image = batch["image"].to(device)
            depth = batch["depth"].to(device)
            seg = batch["track"].to(device)

            optimizer.zero_grad()
            seg_logits, depth_pred = model(image)
            loss_seg = seg_loss_fn(seg_logits, seg)
            loss_depth = depth_loss_fn(depth_pred, depth)
            loss = loss_seg + loss_depth
            loss.backward()
            optimizer.step()

            seg_cm.update(torch.argmax(seg_logits, dim=1), seg)
            total_depth_loss += loss_depth.item()

        train_miou = seg_cm.mean_iou().item()
        logger.add_scalar("train/miou", train_miou, epoch)
        logger.add_scalar("train/depth_loss", total_depth_loss / len(train_loader), epoch)

        # Validation
        model.eval()
        seg_cm.reset()
        val_depth_loss = 0.0
        with torch.inference_mode():
            for batch in val_loader:
                image = batch["image"].to(device)
                depth = batch["depth"].to(device)
                seg = batch["track"].to(device)

                seg_logits, depth_pred = model(image)
                seg_cm.update(torch.argmax(seg_logits, dim=1), seg)
                val_depth_loss += depth_loss_fn(depth_pred, depth).item()

        val_miou = seg_cm.mean_iou().item()
        logger.add_scalar("val/miou", val_miou, epoch)
        logger.add_scalar("val/depth_loss", val_depth_loss / len(val_loader), epoch)

        print(f"[Epoch {epoch}] mIoU: {val_miou:.4f}, Depth MAE: {val_depth_loss / len(val_loader):.4f}")

        if val_miou > best_miou:
            best_miou = val_miou
            save_model(log_dir, model)

            fixed_model_dir = Path(exp_dir) / model_name
            fixed_model_dir.mkdir(parents=True, exist_ok=True)
            save_model(fixed_model_dir, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default="logs")
    parser.add_argument("--model_name", default="detector")
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(**vars(args))
