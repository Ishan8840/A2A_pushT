"""Convert LeRobot v3 parquet/video data to a timestamp-aligned NumPy cache."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download


def prepare(output, revision="main"):
    output = Path(output)
    if (output / "manifest.json").exists():
        print(f"Prepared cache already exists: {output}", flush=True)
        return
    sha = HfApi().dataset_info("lerobot/pusht", revision=revision).sha
    source = Path(snapshot_download("lerobot/pusht", repo_type="dataset", revision=sha,
                                   allow_patterns=["meta/*", "data/*", "videos/*"]))
    info = json.loads((source / "meta/info.json").read_text())
    if info["codebase_version"] != "v3.0":
        raise ValueError("This converter expects LeRobot v3.0; supply a v3 revision.")
    episodes = []
    for path in sorted((source / "meta/episodes").rglob("*.parquet")):
        episodes.extend(pq.read_table(path).to_pylist())
    episodes.sort(key=lambda x: x["episode_index"])
    rows = []
    for path in sorted((source / "data").rglob("*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    by_episode = defaultdict(list)
    for row in rows:
        by_episode[row["episode_index"]].append(row)
    assert len(episodes) == info["total_episodes"]
    assert len(rows) == info["total_frames"]
    output.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    images = np.lib.format.open_memmap(output / "images.npy", mode="w+", dtype="uint8",
                                     shape=(n, 96, 96, 3))
    states = np.empty((n, 2), dtype=np.float32)
    actions = np.empty((n, 2), dtype=np.float32)
    jobs = defaultdict(list)
    bounds, cursor = [], 0
    key = "observation.image"
    for ep in episodes:
        episode_rows = sorted(by_episode[ep["episode_index"]], key=lambda r: r["frame_index"])
        assert len(episode_rows) == ep["length"]
        assert [r["frame_index"] for r in episode_rows] == list(range(ep["length"]))
        video = info["video_path"].format(video_key=key,
            chunk_index=ep[f"videos/{key}/chunk_index"],
            file_index=ep[f"videos/{key}/file_index"])
        offset = ep[f"videos/{key}/from_timestamp"]
        start = cursor
        for row in episode_rows:
            states[cursor] = row["observation.state"]
            actions[cursor] = row["action"]
            jobs[video].append((offset + row["timestamp"], cursor))
            cursor += 1
        bounds.append([start, cursor])
    for video, requests in jobs.items():
        requests.sort()
        pos, previous = 0, None
        with av.open(str(source / video)) as container:
            for frame in container.decode(video=0):
                timestamp = float(frame.pts * frame.time_base)
                rgb = frame.to_ndarray(format="rgb24")
                assert rgb.shape == (96, 96, 3)
                while pos < len(requests) and requests[pos][0] <= timestamp + 1e-6:
                    target, idx = requests[pos]
                    chosen = (timestamp, rgb)
                    if previous is not None and abs(previous[0] - target) < abs(timestamp - target):
                        chosen = previous
                    if abs(chosen[0] - target) > 0.051:
                        raise ValueError(f"Video alignment error: {video}, {target}, {chosen[0]}")
                    images[idx] = chosen[1]
                    pos += 1
                previous = (timestamp, rgb)
            while pos < len(requests):
                target, idx = requests[pos]
                if previous is None or abs(previous[0] - target) > 0.051:
                    raise ValueError(f"Missing video frame: {video}, {target}")
                images[idx] = previous[1]
                pos += 1
        print(f"Aligned {len(requests)} frames from {video}", flush=True)
    images.flush()
    np.save(output / "states.npy", states)
    np.save(output / "actions.npy", actions)
    np.save(output / "episode_bounds.npy", np.array(bounds, dtype=np.int64))
    manifest = dict(repo_id="lerobot/pusht", revision=sha, fps=info["fps"],
                    frames=n, episodes=len(bounds), format="a2a-pusht-numpy-v1")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/pusht")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    prepare(args.output, args.revision)
