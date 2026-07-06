"""Device-model lidar: XT16 preset geometry + SimulatedLidar imperfections."""

import torch

from domo.robot.lidar_models import (
    XT16_FULL_AZIMUTH,
    LidarModelConfig,
    SimulatedLidar,
    generic_sector_lidar,
    hesai_xt16,
)

N_ENVS, N_SECTORS = 4, 36


class FakeHandle:
    """Returns a constant range field set by the test."""

    def __init__(self, model: LidarModelConfig, fill: float):
        self.config = model.to_lidar_config()
        self._ranges = torch.full(
            (N_ENVS, model.n_vertical, model.n_horizontal), fill)

    def set(self, ranges):
        self._ranges = ranges

    def read_ranges(self):
        return self._ranges.clamp(0.0, self.config.max_range)

    def read_sector_distances(self):
        return self._ranges.min(dim=1).values


def _sensor(model, fill=2.0):
    handle = FakeHandle(model, fill)
    return handle, SimulatedLidar(handle, model, N_ENVS, N_SECTORS,
                                  torch.device("cpu"), control_dt=0.02)


def _scan(sensor):
    for _ in range(sensor.update_interval):
        sensor.tick()


def test_xt16_preset_geometry():
    m = hesai_xt16()
    assert m.n_vertical == 16
    assert m.fov_deg == (360.0, 30.0)          # ±15°, 2° channel spacing
    assert m.max_range == 120.0 and m.min_range == 0.05
    assert m.update_interval(0.02) == 5        # 10 Hz at 50 Hz control
    assert hesai_xt16(rate_hz=20.0).update_interval(0.02) == 2  # rounded
    assert XT16_FULL_AZIMUTH % N_SECTORS == 0  # full res pools exactly


def test_scan_rate_caching():
    m = hesai_xt16(n_horizontal=72)
    m.range_noise_std = 0.0
    m.dropout_prob = 0.0
    handle, sensor = _sensor(m, fill=2.0)

    assert not sensor.has_scan
    sensor.tick()                               # 1 of 5 — no scan yet
    assert (sensor.read() == m.max_range).all()
    for _ in range(4):
        sensor.tick()                           # frame due at step 5
    assert sensor.has_scan
    assert torch.allclose(sensor.read(), torch.full((N_ENVS, N_SECTORS), 2.0))

    handle.set(torch.full((N_ENVS, 16, 72), 1.0))
    sensor.tick()                               # between frames: cached
    assert torch.allclose(sensor.read(), torch.full((N_ENVS, N_SECTORS), 2.0))


def test_sector_min_pooling():
    m = generic_sector_lidar()                  # 36 × 5, noiseless
    handle, sensor = _sensor(m, fill=3.0)
    ranges = torch.full((N_ENVS, 5, 36), 3.0)
    ranges[:, 2, 7] = 0.8                       # one close return in sector 7
    handle.set(ranges)
    _scan(sensor)
    sectors = sensor.read()
    assert torch.isclose(sectors[0, 7], torch.tensor(0.8))
    assert torch.isclose(sectors[0, 6], torch.tensor(3.0))


def test_blind_zone_reports_max_range():
    m = hesai_xt16(n_horizontal=72)
    m.range_noise_std = 0.0
    m.dropout_prob = 0.0
    handle, sensor = _sensor(m, fill=2.0)
    ranges = torch.full((N_ENVS, 16, 72), 2.0)
    ranges[:, :, 0:2] = 0.01                    # sector 0 (2 az cols) in blind zone
    handle.set(ranges)
    _scan(sensor)
    assert (sensor.read()[:, 0] == m.max_range).all()
    assert torch.allclose(sensor.read()[:, 1], torch.full((N_ENVS,), 2.0))


def test_noise_statistics():
    torch.manual_seed(0)
    m = hesai_xt16(n_horizontal=720)
    m.dropout_prob = 0.0
    handle, sensor = _sensor(m, fill=2.0)
    _scan(sensor)
    raw = sensor.read_raw()
    assert abs(raw.mean().item() - 2.0) < 0.005
    assert abs(raw.std().item() - m.range_noise_std) < 0.002


def test_dropout_fraction():
    torch.manual_seed(0)
    m = hesai_xt16(n_horizontal=720)
    m.range_noise_std = 0.0
    m.dropout_prob = 0.05
    handle, sensor = _sensor(m, fill=2.0)
    _scan(sensor)
    frac = (sensor.read_raw() == m.max_range).float().mean().item()
    assert abs(frac - 0.05) < 0.01


def test_reset_idx():
    m = generic_sector_lidar()
    handle, sensor = _sensor(m, fill=1.0)
    _scan(sensor)
    assert (sensor.read() == 1.0).all()
    sensor.reset_idx(torch.tensor([0, 2]))
    assert (sensor.read()[[0, 2]] == m.max_range).all()
    assert (sensor.read()[[1, 3]] == 1.0).all()


def test_rejects_non_divisible_azimuth():
    m = hesai_xt16(n_horizontal=100)            # 100 % 36 != 0
    try:
        _sensor(m)
        assert False, "expected ValueError"
    except ValueError:
        pass
