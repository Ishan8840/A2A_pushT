import argparse
import copy
import json
import math
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from pusht_a2a.data import PushTDataset, split_episodes
from pusht_a2a.model import build_policy
from roboverse_learn.il.utils.pytorch_util import dict_apply


def git_sha(path=None):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pusht.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int, help="Limit optimizer steps for a smoke run")
    parser.add_argument("overrides", nargs="*", help="OmegaConf key=value overrides")
    args = parser.parse_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run training on the GPU server")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    output = Path(cfg.output_dir)
    if (output / "last.pt").exists() and not args.resume:
        raise FileExistsError("Output has a checkpoint; use --resume or a new output_dir")
    output.mkdir(parents=True, exist_ok=True)
    bounds = np.load(Path(cfg.data_dir) / "episode_bounds.npy")
    train_ids, val_ids = split_episodes(len(bounds), cfg.val_ratio, cfg.seed)
    train = PushTDataset(cfg.data_dir, train_ids, cfg.n_obs_steps, cfg.n_action_steps)
    val = PushTDataset(cfg.data_dir, val_ids, cfg.n_obs_steps, cfg.n_action_steps)
    generator = torch.Generator().manual_seed(cfg.seed)
    loader_kwargs = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                         pin_memory=cfg.device == "cuda", persistent_workers=cfg.num_workers > 0)
    train_loader = DataLoader(train, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val, shuffle=False, **loader_kwargs)
    policy = build_policy(cfg)
    policy.set_normalizer(train.normalizer())
    policy.to(cfg.device)
    ema = copy.deepcopy(policy).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],
                                  lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.epochs

    def lr_factor(step):
        if step < cfg.warmup_steps:
            return (step + 1) / max(1, cfg.warmup_steps)
        progress = min(1, (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    start_epoch, step, best = 0, 0, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("partial_epoch"):
            raise ValueError("Smoke checkpoints cannot resume exactly; use a complete epoch checkpoint")
        old = checkpoint["config"]
        current = OmegaConf.to_container(cfg, resolve=True)
        for key in current:
            if key not in ("output_dir", "device", "num_workers") and current[key] != old[key]:
                raise ValueError(f"Resume configuration differs for {key}; keep the original schedule")
        if checkpoint["dataset"] != train.manifest:
            raise ValueError("Resume dataset revision differs")
        policy.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch, step, best = checkpoint["epoch"] + 1, checkpoint["step"], checkpoint["best_val"]
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        generator.set_state(checkpoint["loader_rng"])
    provenance = dict(config=OmegaConf.to_container(cfg, resolve=True), dataset=train.manifest,
                      git_sha=git_sha(), upstream_sha=git_sha(__import__("pusht_a2a").UPSTREAM),
                      train_episodes=train_ids.tolist(), val_episodes=val_ids.tolist(),
                      torch=torch.__version__, cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(0) if cfg.device == "cuda" else "cpu")
    (output / "run.json").write_text(json.dumps(provenance, indent=2) + "\n")
    OmegaConf.save(cfg, output / "config.yaml")
    print(json.dumps(provenance, indent=2), flush=True)
    for epoch in range(start_epoch, cfg.epochs):
        policy.train()
        train_sum, train_count = 0., 0
        limited = False
        for batch in train_loader:
            batch = dict_apply(batch, lambda x: x.to(cfg.device, non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            loss = policy.compute_loss(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip,
                                                      error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                for target, source in zip(ema.parameters(), policy.parameters()):
                    target.lerp_(source, 1 - cfg.ema_decay)
                for target, source in zip(ema.buffers(), policy.buffers()):
                    target.copy_(source)
            count = batch["action"].shape[0]
            train_sum += loss.item() * count
            train_count += count
            step += 1
            if step % 50 == 0 or step <= 3:
                print(json.dumps(dict(step=step, epoch=epoch, loss=loss.item(),
                                      grad_norm=float(grad_norm), lr=scheduler.get_last_lr()[0])), flush=True)
            if args.max_steps and step >= args.max_steps:
                limited = True
                break
        ema.eval()
        val_sum, val_count = 0., 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                batch = dict_apply(batch, lambda x: x.to(cfg.device, non_blocking=True))
                loss = ema.compute_loss(batch)
                count = batch["action"].shape[0]
                val_sum += loss.item() * count
                val_count += count
                if limited:
                    break
        val_loss = val_sum / val_count
        if not math.isfinite(val_loss):
            raise RuntimeError("Non-finite validation loss")
        improved = val_loss < best
        best = min(best, val_loss)
        metrics = dict(epoch=epoch, step=step, train_loss=train_sum / train_count, val_loss=val_loss)
        if not limited and cfg.eval_every > 0 and (epoch + 1) % cfg.eval_every == 0:
            from pusht_a2a.evaluate import evaluate
            report = evaluate(ema, cfg, cfg.eval_episodes, cfg.eval_seed,
                              output / f"eval/epoch_{epoch + 1:04d}")
            metrics.update(success_rate=report["success_rate"], mean_max_reward=report["mean_max_reward"])
        checkpoint = dict(model=policy.state_dict(), ema=ema.state_dict(),
                          optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                          config=provenance["config"], dataset=train.manifest, epoch=epoch,
                          step=step, best_val=best, torch_rng=torch.get_rng_state(),
                          cuda_rng=torch.cuda.get_rng_state_all(), numpy_rng=np.random.get_state(),
                          python_rng=random.getstate(), loader_rng=generator.get_state(),
                          partial_epoch=limited)
        # A smoke checkpoint is usable for evaluation, but is not an exact epoch resume.
        temporary = output / "last.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(output / "last.pt")
        if improved:
            torch.save(checkpoint, output / "best.pt")
        if (epoch + 1) % cfg.checkpoint_every == 0:
            torch.save(checkpoint, output / f"epoch_{epoch + 1:04d}.pt")
        with (output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)
        if limited:
            break


if __name__ == "__main__":
    main()
