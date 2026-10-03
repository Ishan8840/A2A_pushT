# A2A on PushT

Train the original [A2A flow matching policy](https://github.com/JIAjindou/A2A_Flow_Matching)
on [lerobot/pusht](https://huggingface.co/datasets/lerobot/pusht) and evaluate in
[gym-pusht](https://github.com/huggingface/gym-pusht). The upstream implementation is
a Git submodule pinned to `a5792ecf4e7f8fa4d85fe66ea9a50618138f925c`.
No Isaac Sim or RoboVerse simulator installation is needed.

All installs, data preparation, tests, training, and evaluation run on the GPU server.
The laptop is only for source editing and Git operations.

## Server setup

```bash
ssh -p 32033 root@202.122.49.242
cd /workspace
git clone --recurse-submodules https://github.com/Ishan8840/A2A_pushT.git
cd A2A_pushT
uv venv --python /venv/main/bin/python .venv
uv pip install --python .venv/bin/python torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pusht_a2a.prepare
```

Python 3.12 is the tested server Python. The existing server environment uses the
same PyTorch CUDA build. Pymunk is pinned to 6.11.1 because gym-pusht 0.1.6 uses its API.
The prepared cache requires roughly 710 MB, plus the original Hugging Face download.
The converter resolves the dataset revision to a SHA and records it in
`data/pusht/manifest.json`. Use `--revision SHA` and a new `--output` to prepare a
specific version. Existing completed caches are reused.

## Validate, train, evaluate

```bash
.venv/bin/python -m pytest -q
# Short GPU training check with real data; keeps full training outputs separate.
.venv/bin/python -m pusht_a2a.train --max-steps 3 \
  output_dir=outputs/smoke batch_size=4 num_workers=0
.venv/bin/python -m pusht_a2a.evaluate --checkpoint outputs/smoke/last.pt \
  --episodes 1 --max-steps 16 --output outputs/smoke/eval

# Full run: 200 epochs, batch size 64, evaluation every 10 epochs.
.venv/bin/python -m pusht_a2a.train
# Resume a completed epoch with identical training settings.
.venv/bin/python -m pusht_a2a.train --resume outputs/a2a_pusht/last.pt
# Independent evaluation seeds, using EMA weights.
.venv/bin/python -m pusht_a2a.evaluate --checkpoint outputs/a2a_pusht/best.pt \
  --episodes 50 --seed 20000 --output outputs/final_eval
```

Configuration is in `configs/pusht.yaml`; overrides use `key=value`. Outputs contain
`run.json` (data SHA, code SHAs, split, hardware), resolved config, JSONL metrics,
atomic `last.pt`, `best.pt` selected by validation loss, and periodic checkpoints.
Checkpoints include the policy, EMA, normalizer, optimizer, schedule and RNG states.
Smoke checkpoints end partway through an epoch and support evaluation only.
Evaluation writes success rate, maximum coverage reward and a video of the first episode.
Full training uses a cosine learning-rate schedule, warmup, gradient clipping and EMA.
Long runs on this Vast instance should be managed with supervisor; see
`scripts/supervisor-a2a.conf` and `scripts/train-server.sh`.

## Data and policy contract

- Dataset v3 parquet and AV1 video timestamps are aligned before training. Images
  remain 96×96; `observation.state` supplies the two agent position coordinates.
  Actions are absolute `[x, y]` targets in environment coordinates.
- Each sample contains eight images/positions ending at time `t` and a padded
  15-action window. Upstream A2A takes entries 7–14, predicting actions `t` to `t+7`.
  Endpoint padding stays within each episode. Evaluation updates history after
  every environment step and executes eight predicted actions per inference.
- A seeded episode split assigns 185 training and 21 validation episodes.
  State/action normalization is fit only on training episodes. Images remain
  `[0, 1]` before the upstream encoder's ImageNet normalization.
- The original ResNet18 with GroupNorm, latent encoders/decoder, conditional
  flow loss, consistency loss and both reconstruction losses are reused directly.
  Six Euler flow steps map encoded history states to future action latents.
  History and future lengths must be multiples of eight for the upstream CNNs.
- Evaluation uses `pixels_agent_pos`, 10 Hz control, no privileged block state,
  clipping to the environment action space and a 300-step episode limit.

This is a PushT adaptation, not a claim to reproduce a published PushT result.
Full training quality must be measured with the independent evaluation command.
