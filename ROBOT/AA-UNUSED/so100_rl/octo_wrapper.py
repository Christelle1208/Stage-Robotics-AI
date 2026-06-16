"""
Gym wrapper that makes SO100PickPlaceEnv compatible with Octo's expected interface.

Octo expects dict observations:
  - "image_primary"      : (H, W, 3) uint8 — rendered camera frame
  - "proprio"  (optional): (6,) float32    — joint positions (qpos)

It also expects the environment to expose a get_task() method returning a dict
with a "language_instruction" key.

Action space note
-----------------
Octo (pretrained on bridge_dataset) outputs 7-D EEF-delta actions.
SO100 uses 7-D joint-delta actions. The dimensions match by coincidence but the
semantics differ. For zero-shot exploratory runs we pass Octo's output through
directly (clipped to [-1, 1]). To get meaningful behaviour, fine-tune Octo on
SO100 demonstration data collected with `train_so100_pick_place.py`.
"""
from __future__ import annotations

from typing import Optional
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Language instruction defaults for each task
TASK_INSTRUCTIONS: dict[str, str] = {
    "reach": "move the arm to the red cube",
    "pick": "pick up the red cube",
    "pick_place": "pick up the red cube and place it in the green zone",
}


def _resize_image(img: np.ndarray, size: int) -> np.ndarray:
    """Resize h×w×3 uint8 image to size×size using nearest-neighbour (no deps)."""
    h, w = img.shape[:2]
    if h == size and w == size:
        return img
    try:
        import cv2  # type: ignore
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    except ImportError:
        sy = np.linspace(0, h - 1, size, dtype=int)
        sx = np.linspace(0, w - 1, size, dtype=int)
        return img[np.ix_(sy, sx)]


class OctoSO100Wrapper(gym.Wrapper):
    """
    Wraps SO100PickPlaceEnv to expose an Octo-compatible observation interface.

    Parameters
    ----------
    env : SO100PickPlaceEnv
        Must be constructed with render_mode="rgb_array".
    image_size : int
        Square size for the rendered image (Octo default: 256).
    include_proprio : bool
        Whether to add a "proprio" key (6-D joint positions) to the observation.
    language_instruction : str | None
        Override the language instruction. Defaults to a per-task sentence.
    """

    def __init__(
        self,
        env: gym.Env,
        image_size: int = 256,
        include_proprio: bool = True,
        language_instruction: Optional[str] = None,
    ) -> None:
        if getattr(env, "render_mode", None) != "rgb_array":
            raise ValueError(
                "OctoSO100Wrapper requires render_mode='rgb_array'. "
                "Build the env with SO100PickPlaceEnv(..., render_mode='rgb_array')."
            )
        super().__init__(env)
        self.image_size = image_size
        self.include_proprio = include_proprio
        self._lang: str = language_instruction or TASK_INSTRUCTIONS.get(
            getattr(env, "task", "reach"), "manipulate the object"
        )

        # Build new Dict observation space
        img_space = gym.spaces.Box(
            low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
        )
        spaces: dict[str, gym.Space] = {"image_primary": img_space}
        if include_proprio:
            spaces["proprio"] = gym.spaces.Box(
                low=-np.pi, high=np.pi, shape=(6,), dtype=np.float32
            )
        self.observation_space = gym.spaces.Dict(spaces)

    # ------------------------------------------------------------------
    # Octo interface
    # ------------------------------------------------------------------

    def get_task(self) -> dict:
        """Return a task dict suitable for model.create_tasks(texts=...)."""
        return {"language_instruction": self._lang}

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        flat_obs, info = self.env.reset(**kwargs)
        return self._to_dict_obs(flat_obs), info

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        flat_obs, reward, terminated, truncated, info = self.env.step(action)
        return self._to_dict_obs(flat_obs), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_dict_obs(self, flat_obs: np.ndarray) -> dict:
        img = self.env.render()  # (H, W, 3) uint8 — rendered by MuJoCo
        img = _resize_image(img, self.image_size)
        obs: dict = {"image_primary": img}
        if self.include_proprio:
            # First 6 dims of flat obs are qpos
            obs["proprio"] = flat_obs[:6].astype(np.float32)
        return obs
