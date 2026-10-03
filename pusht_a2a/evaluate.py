import argparse
import json
import os
from collections import deque
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import av
import gymnasium as gym
import gym_pusht  # noqa: F401, registers the environment
import numpy as np
import torch
from omegaconf import OmegaConf

from pusht_a2a.model import build_policy


def save_video(path, frames):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height = frames[0].shape[1], frames[0].shape[0]
        stream.pix_fmt = "yuv420p"
        for rgb in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@torch.no_grad()
def evaluate(policy, cfg, episodes, seed, output=None, max_steps=None):
    policy.eval()
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos",
                   render_mode="rgb_array", observation_width=96, observation_height=96,
                   max_episode_steps=max_steps or cfg.max_episode_steps)
    results = []
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            history = deque([obs] * cfg.n_obs_steps, maxlen=cfg.n_obs_steps)
            frames = [obs["pixels"]] if output is not None and episode == 0 else None
            rewards, success, done = [], False, False
            while not done:
                images = np.stack([o["pixels"] for o in history])
                positions = np.stack([o["agent_pos"] for o in history])
                batch = {"head_cam": torch.from_numpy(images).permute(0, 3, 1, 2)
                         .float().unsqueeze(0).to(policy.device) / 255,
                         "agent_pos": torch.from_numpy(positions).float().unsqueeze(0).to(policy.device)}
                actions = policy.predict_action(batch)["action"][0].cpu().numpy()
                if not np.isfinite(actions).all():
                    raise RuntimeError("Policy predicted non-finite actions")
                for action in actions:
                    obs, reward, terminated, truncated, info = env.step(
                        np.clip(action, env.action_space.low, env.action_space.high))
                    history.append(obs)
                    rewards.append(float(reward))
                    success |= bool(info.get("is_success", terminated))
                    if frames is not None:
                        frames.append(obs["pixels"])
                    done = terminated or truncated
                    if done:
                        break
            results.append(dict(seed=seed + episode, success=success, steps=len(rewards),
                                max_reward=max(rewards), reward_sum=sum(rewards)))
            if frames is not None:
                Path(output).mkdir(parents=True, exist_ok=True)
                save_video(Path(output) / "episode_000.mp4", frames)
    finally:
        env.close()
    report = dict(success_rate=float(np.mean([r["success"] for r in results])),
                  mean_max_reward=float(np.mean([r["max_reward"] for r in results])),
                  episodes=results)
    if output is not None:
        (Path(output) / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="outputs/eval")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(checkpoint["config"])
    policy = build_policy(cfg)
    policy.load_state_dict(checkpoint.get("ema", checkpoint["model"]))
    policy.to(args.device)
    report = evaluate(policy, cfg, args.episodes, args.seed, args.output, args.max_steps)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
