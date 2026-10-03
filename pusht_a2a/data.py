"""Episode-safe windows: history ends at t; future actions start at t."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from roboverse_learn.il.utils.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


def split_episodes(count, val_ratio, seed):
    if not 0 < val_ratio < 1 or count < 2:
        raise ValueError("Need at least two episodes and 0 < val_ratio < 1")
    order = np.random.default_rng(seed).permutation(count)
    n_val = min(count - 1, max(1, round(count * val_ratio)))
    return np.sort(order[n_val:]), np.sort(order[:n_val])


class PushTDataset(Dataset):
    def __init__(self, root, episodes, n_obs_steps=8, n_action_steps=8):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["fps"] != 10:
            raise ValueError("PushT evaluation assumes a 10 Hz dataset")
        self.images = np.load(self.root / "images.npy", mmap_mode="r")
        self.states = np.load(self.root / "states.npy", mmap_mode="r")
        self.actions = np.load(self.root / "actions.npy", mmap_mode="r")
        self.bounds = np.load(self.root / "episode_bounds.npy")
        self.episodes = np.asarray(episodes, dtype=np.int64)
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.indices = [(int(start), int(end), t) for start, end in self.bounds[self.episodes]
                        for t in range(int(start), int(end))]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        start, end, t = self.indices[index]
        history = np.clip(np.arange(t - self.n_obs_steps + 1, t + 1), start, end - 1)
        # Upstream compute_loss slices actions[n_obs_steps-1:...].
        action_idx = np.clip(np.arange(t - self.n_obs_steps + 1, t + self.n_action_steps),
                             start, end - 1)
        return {"obs": {
            "head_cam": torch.from_numpy(self.images[history].copy()).permute(0, 3, 1, 2).float() / 255,
            "agent_pos": torch.from_numpy(self.states[history].copy())},
            "action": torch.from_numpy(self.actions[action_idx].copy())}

    def normalizer(self):
        selected = np.concatenate([np.arange(s, e) for s, e in self.bounds[self.episodes]])
        normalizer = LinearNormalizer()
        normalizer.fit({"agent_pos": self.states[selected], "action": self.actions[selected]})
        # The upstream vision encoder applies ImageNet normalization itself.
        normalizer["head_cam"] = SingleFieldLinearNormalizer.create_identity()
        return normalizer
