"""
Custom feature extractor for the SO-100 pick-and-place PPO policies.

Both PickEnv and PlaceEnv use flat Box observations (state vectors only,
no images), so we replace the CNN-based extractor with a lightweight MLP
that is fast to train and sufficient for proprioceptive control.

  PickExtractor  : input = 18-dim flat vector
  PlaceExtractor : input = 23-dim flat vector

V3 extractors:
  ReachExtractor  : alias of PickExtractor   (18-dim)
  GraspExtractor  : same architecture        (18-dim)
  CarryExtractor  : same architecture        (21-dim)
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PickExtractor(BaseFeaturesExtractor):
    """
    MLP feature extractor for PickEnv (obs_dim = 19).
    Architecture: 19 → 128 → 256 → features_dim
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        obs_dim = observation_space.shape[0]  # 19
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


# ---------------------------------------------------------------------------
# V3 extractors
# ---------------------------------------------------------------------------
# ReachEnv and GraspEnv share the same 18-dim obs layout as PickEnv.
ReachExtractor = PickExtractor  # alias — identical architecture


class GraspExtractor(BaseFeaturesExtractor):
    """MLP extractor for GraspEnv (obs_dim = 18)."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        obs_dim = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 256),    nn.ReLU(),
            nn.Linear(256, features_dim), nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class CarryExtractor(BaseFeaturesExtractor):
    """MLP extractor for CarryEnv (obs_dim = 21, includes goal direction)."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        obs_dim = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 256),    nn.ReLU(),
            nn.Linear(256, features_dim), nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class PlaceExtractor(BaseFeaturesExtractor):
    """
    MLP feature extractor for PlaceEnv (obs_dim = 22).
    Architecture: 22 → 128 → 256 → features_dim
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        obs_dim = observation_space.shape[0]  # 22
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)