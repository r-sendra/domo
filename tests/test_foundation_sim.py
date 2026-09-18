"""
Engine-free checks for domo.sim: the backend registry and the lidar layout
helper backends use to honour the [N, n_vertical, n_horizontal] contract.
"""

import pytest
import torch

from domo.sim import PhysicsEngine, create_engine, register_backend
from domo.sim.base import lidar_ranges_to_grid


class _StubEngine(PhysicsEngine):
    name = "stub"

    def __init__(self, device="cpu"):
        self._device = torch.device(device)

    def create_scene(self, cfg):
        raise NotImplementedError

    @property
    def device(self):
        return self._device


def test_registry_creates_registered_backend_with_kwargs():
    register_backend("stub-test", lambda **kw: _StubEngine(**kw))
    engine = create_engine("stub-test", device="cpu")
    assert isinstance(engine, _StubEngine)
    assert engine.device == torch.device("cpu")


def test_registry_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown physics backend"):
        create_engine("no-such-engine")


def test_sim_package_exports_camera_handle():
    from domo.sim import CameraHandle
    assert CameraHandle.__name__ == "CameraHandle"


# ---------------------------------------------------------------------------
# lidar_ranges_to_grid
# ---------------------------------------------------------------------------

N_V, N_H = 5, 36


def _azimuth_major_grid(n_env=2):
    """raw[e, h, v] = h + 100*v, so azimuth and elevation are identifiable."""
    h = torch.arange(N_H, dtype=torch.float32).view(1, N_H, 1)
    v = torch.arange(N_V, dtype=torch.float32).view(1, 1, N_V)
    return (h + 100 * v).expand(n_env, N_H, N_V).clone()


def test_azimuth_major_3d_is_transposed_to_contract_layout():
    raw = _azimuth_major_grid()
    grid = lidar_ranges_to_grid(raw, N_V, N_H)
    assert grid.shape == (2, N_V, N_H)
    # grid[e, v, h] must be h + 100*v: rows are elevation channels.
    assert torch.equal(grid[0, 2], torch.arange(N_H, dtype=torch.float32) + 200)
    assert torch.equal(grid[1, :, 7], 7 + 100 * torch.arange(N_V, dtype=torch.float32))


def test_contract_layout_3d_passes_through():
    raw = _azimuth_major_grid().transpose(1, 2)          # already [N, v, h]
    assert torch.equal(lidar_ranges_to_grid(raw, N_V, N_H), raw)


def test_flat_buffer_respects_native_order():
    raw = _azimuth_major_grid()
    flat = raw.reshape(2, -1)
    assert torch.equal(lidar_ranges_to_grid(flat, N_V, N_H, azimuth_major=True),
                       raw.transpose(1, 2))
    flat_elev_major = raw.transpose(1, 2).reshape(2, -1)
    assert torch.equal(
        lidar_ranges_to_grid(flat_elev_major, N_V, N_H, azimuth_major=False),
        raw.transpose(1, 2))


def test_sector_minimum_matches_script_pooling():
    # The original avoidance scripts grouped contiguous rays of the Genesis
    # buffer (one azimuth's vertical rays); the contract grid must agree.
    raw = torch.rand(3, N_H, N_V)
    expected = raw.reshape(3, -1).view(3, N_H, N_V).min(dim=2).values
    got = lidar_ranges_to_grid(raw, N_V, N_H).min(dim=1).values
    assert torch.allclose(got, expected)


def test_square_grid_uses_azimuth_major_hint():
    raw = torch.arange(2 * 4 * 4, dtype=torch.float32).view(2, 4, 4)
    assert torch.equal(lidar_ranges_to_grid(raw, 4, 4, azimuth_major=True),
                       raw.transpose(1, 2))
    assert torch.equal(lidar_ranges_to_grid(raw, 4, 4, azimuth_major=False), raw)


def test_wrong_beam_count_raises():
    with pytest.raises(RuntimeError, match="expected 5x36"):
        lidar_ranges_to_grid(torch.zeros(2, 7, 9), N_V, N_H)
