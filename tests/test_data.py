import json

import numpy as np
import torch

from pusht_a2a.data import PushTDataset, split_episodes


def make_dataset(tmp_path):
    # Deliberately different coordinate ranges in two episodes.
    state = np.array([[10, 20], [20, 30], [30, 40], [400, 410], [450, 460]], dtype=np.float32)
    np.save(tmp_path / "states.npy", state)
    np.save(tmp_path / "actions.npy", state + 1)
    np.save(tmp_path / "images.npy", np.zeros((5, 96, 96, 3), dtype=np.uint8))
    np.save(tmp_path / "episode_bounds.npy", np.array([[0, 3], [3, 5]]))
    (tmp_path / "manifest.json").write_text(json.dumps({"fps": 10}))
    return PushTDataset(tmp_path, [0, 1])


def test_history_and_future_alignment_and_episode_padding(tmp_path):
    data = make_dataset(tmp_path)
    first = data[0]
    torch.testing.assert_close(first["obs"]["agent_pos"], torch.tensor([[10., 20.]]).repeat(8, 1))
    # Current action begins at index 7, followed by future actions and endpoint padding.
    torch.testing.assert_close(first["action"][7:10], torch.tensor([[11., 21.], [21., 31.], [31., 41.]]))
    last = data[2]
    assert (last["action"][7:] == torch.tensor([31., 41.])).all()
    second = data[3]
    assert (second["obs"]["agent_pos"] == torch.tensor([400., 410.])).all()
    assert last["obs"]["head_cam"].shape == (8, 3, 96, 96)
    assert last["action"].shape == (15, 2)


def test_normalizer_uses_only_train_episodes_and_keeps_image_range(tmp_path):
    make_dataset(tmp_path)
    train = PushTDataset(tmp_path, [0])
    norm = train.normalizer()
    torch.testing.assert_close(norm["agent_pos"].normalize(torch.tensor([30., 40.])), torch.ones(2))
    assert (norm["agent_pos"].normalize(torch.tensor([400., 410.])) > 1).all()
    image = torch.rand(2, 3, 96, 96)
    torch.testing.assert_close(norm["head_cam"].normalize(image), image)


def test_split_is_disjoint_reproducible_and_exhaustive():
    train, val = split_episodes(206, 0.1, 42)
    assert len(train) == 185 and len(val) == 21
    assert not set(train) & set(val)
    assert set(train) | set(val) == set(range(206))
    assert np.array_equal(train, split_episodes(206, 0.1, 42)[0])
