from pathlib import Path

import torch
import torch.nn as nn


HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints



        hidden_dim = 128
        input_dim = 2 * n_track * 2  # left and right track points (x, y)
        output_dim = n_waypoints * 2  # waypoints (x, y)

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        #raise NotImplementedError
        x = torch.cat([track_left, track_right], dim=2)  
        waypoints = self.mlp(x).view(-1, self.n_waypoints, 2)  # reshape to (b, n_waypoints, 2)
        return waypoints


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Use centerline only
        self.encoder = nn.Linear(2, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, n_track, d_model))

        # Small decoder
        self.query_embed = nn.Embedding(n_waypoints, d_model)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=2, dim_feedforward=64),
            num_layers=1,
        )
        self.fc_out = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        #raise NotImplementedError

        b = track_left.size(0)

        # 1. Build and normalize centerline
        centerline = 0.5 * (track_left + track_right)
        centerline = centerline - centerline.mean(dim=1, keepdim=True)

        # 2. Encode + positional
        memory = self.encoder(centerline) + self.positional_encoding[:, :centerline.shape[1], :]
        memory = memory.permute(1, 0, 2)  # (seq_len, B, d_model)

        # 3. Query
        query = self.query_embed.weight.unsqueeze(1).repeat(1, b, 1)  # (n_waypoints, B, d_model)

        # 4. Decode + project
        output = self.decoder(query, memory)
        output = self.fc_out(output).permute(1, 0, 2)  # (B, n_waypoints, 2)

        return output


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # Lighter CNN Backbone with Global Average Pooling
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # (B, 32, 96, 128)
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2),  # (B, 32, 48, 64)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B, 64, 48, 64)
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(2),  # (B, 64, 24, 32)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (B, 128, 24, 32)
            nn.LeakyReLU(negative_slope=0.1),
            nn.AdaptiveAvgPool2d((1, 1)),  # → (B, 128, 1, 1)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),               # → (B, 128)
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(64, n_waypoints * 2),
        )


    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        x = self.cnn(x)
        x = self.fc(x)
        return x.view(-1, self.n_waypoints, 2)


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
