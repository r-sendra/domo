"""
Commercial lidar device models.

A `LidarModelConfig` describes a physical device (beam geometry, scan rate,
range limits, noise) plus one sim-fidelity knob: `n_horizontal`, the number
of azimuth samples raycast in simulation. Because policies consume sector
minima (see `SimulatedLidar.read`), azimuth sampling changes fidelity but
not the policy interface — train cheap (e.g. 180), validate dense (2000).

`SimulatedLidar` wraps a raw `domo.sim` lidar handle and applies the device
imperfections each scan:
  * blind zone: returns closer than `min_range` become no-returns
  * Gaussian range noise (`range_noise_std`)
  * random dropout (`dropout_prob`) — dark/specular surfaces on the real unit
No-returns are reported as `max_range`, which is how they appear to
downstream occupancy logic on the real driver. These knobs double as a
domain-randomization surface.

Presets:
  * `hesai_xt16()` — Hesai PandarXT-16, the unit mounted on the real Go2:
    16 channels over ±15° (2° spacing), 360° azimuth (0.18° @ 10 Hz →
    2000 pts/ring), 0.05–120 m, σ ≈ 1 cm.
  * `generic_sector_lidar()` — the idealised 36×5 sensor of the original
    avoidance scripts (50° vertical FOV, 4 m, noiseless), kept as default
    for faithful replication of those experiments.

Randomness (noise, dropout) is drawn from torch's global RNG, so seed with
`torch.manual_seed` for reproducible scans.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from domo.sim.base import LidarConfig, LidarSensorHandle

from .sensors import ExteroceptiveSensor

__all__ = [
    "XT16_FULL_AZIMUTH",
    "LidarModelConfig",
    "SimulatedLidar",
    "generic_sector_lidar",
    "hesai_xt16",
]

# The real device scans 2000 pts/ring (0.18° @ 10 Hz). 2000 is not divisible
# by the 36 policy sectors, so full-fidelity sim uses 1980 (0.182°/step,
# 55 rays per sector) — an indistinguishable approximation that pools exactly.
XT16_FULL_AZIMUTH = 1980


@dataclass
class LidarModelConfig:
    """
    Physical lidar description + sim-fidelity knob.

    Ranges in metres, rate in Hz, FOV in degrees (horizontal, vertical).
    `to_lidar_config()` is what the backend receives (geometry only);
    the remaining fields are applied by `SimulatedLidar`.
    """
    name: str = "generic"
    # Beam geometry (n_horizontal is the sim-fidelity knob)
    n_horizontal: int = 36
    n_vertical: int = 5
    fov_deg: tuple[float, float] = (360.0, 50.0)
    # Device behaviour
    rate_hz: float = 10.0
    min_range: float = 0.0
    max_range: float = 4.0
    range_noise_std: float = 0.0
    dropout_prob: float = 0.0
    # Mounting (position offset from the base link; measure on the real robot)
    pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.35)
    draw_debug: bool = False

    def to_lidar_config(self) -> LidarConfig:
        """Geometry-only view handed to `Scene.add_lidar`."""
        return LidarConfig(
            n_horizontal=self.n_horizontal, n_vertical=self.n_vertical,
            fov_deg=self.fov_deg, max_range=self.max_range,
            pos_offset=self.pos_offset, draw_debug=self.draw_debug)

    def update_interval(self, control_dt: float) -> int:
        """Control steps between scans for this device's frame rate (>= 1, rounded)."""
        return max(1, round(1.0 / (self.rate_hz * control_dt)))


def hesai_xt16(n_horizontal: int = 180,
               pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.35),
               rate_hz: float = 10.0,
               draw_debug: bool = False) -> LidarModelConfig:
    """
    Hesai PandarXT-16. `n_horizontal` downsamples the 2000-point ring for
    vectorised training (use XT16_FULL_AZIMUTH for full-fidelity eval).
    Supported real frame rates: 5 / 10 / 20 Hz.
    """
    return LidarModelConfig(
        name="hesai_xt16",
        n_horizontal=n_horizontal,
        n_vertical=16,
        fov_deg=(360.0, 30.0),          # ±15°, uniform 2° channel spacing
        rate_hz=rate_hz,
        min_range=0.05,                 # blind zone
        max_range=120.0,
        range_noise_std=0.01,           # ±1 cm typical accuracy
        dropout_prob=0.003,
        pos_offset=pos_offset,
        draw_debug=draw_debug,
    )


def generic_sector_lidar(draw_debug: bool = False) -> LidarModelConfig:
    """The idealised sensor of the original avoidance scripts (noiseless)."""
    return LidarModelConfig(draw_debug=draw_debug)


class SimulatedLidar(ExteroceptiveSensor):
    """
    Device-model lidar: raw beams + imperfections + sector pooling.

    read()        → [N, n_sectors] azimuth-sector minima (policy-facing; this
                    exact pooling runs on the real driver's point cloud too)
    read_raw()    → [N, n_vertical, n_horizontal] last processed scan
    read_points() → world-frame hit points + validity mask (3D mapping)
    tick()        → advance one control step; rescan when the frame is due
    reset_idx()   → forget the scans of the given envs

    Args:
        handle: backend lidar handle (`Scene.add_lidar`).
        model: device description; `model.n_horizontal` must be a multiple
            of `n_sectors` so every sector pools the same number of rays.
        n_envs / device: batch size and device of the cached tensors.
        n_sectors: azimuth sectors exposed by `read()`.
        control_dt: control step (s), sets the scan interval from `model.rate_hz`.

    Raises:
        ValueError: if `n_horizontal` is not a multiple of `n_sectors`.
    """

    def __init__(self, handle: LidarSensorHandle, model: LidarModelConfig,
                 n_envs: int, n_sectors: int, device: torch.device,
                 control_dt: float):
        if model.n_horizontal % n_sectors != 0:
            raise ValueError(
                f"n_horizontal ({model.n_horizontal}) must be a multiple of "
                f"n_sectors ({n_sectors}) for exact sector pooling")
        self._handle = handle
        self.model = model
        self.n_sectors = n_sectors
        self.max_range = model.max_range
        self._interval = model.update_interval(control_dt)
        self._step_count = 0
        self._raw = torch.full(
            (n_envs, model.n_vertical, model.n_horizontal),
            model.max_range, device=device)
        self._sectors = torch.full(
            (n_envs, n_sectors), model.max_range, device=device)
        # Per-beam world-frame hit points + validity (populated if the sim
        # backend exposes them; enables 3D reconstruction / dense mapping).
        n_beams = model.n_vertical * model.n_horizontal
        self._points = torch.zeros((n_envs, n_beams, 3), device=device)
        self._points_valid = torch.zeros((n_envs, n_beams), dtype=torch.bool,
                                         device=device)
        # Only backends that implement ``read_points`` feed the cloud; fakes
        # without the method and ranges-only handles (which raise
        # NotImplementedError on first use) fall back to sector mode.
        self._has_points = hasattr(handle, "read_points")

    @property
    def update_interval(self) -> int:
        """Control steps between scans."""
        return self._interval

    @property
    def has_scan(self) -> bool:
        """True once at least one scan has been taken."""
        return self._step_count >= self._interval

    def tick(self) -> None:
        """Advance one control step; take a new scan when the frame is due."""
        self._step_count += 1
        if self._step_count % self._interval != 0:
            return
        self._raw = self._apply_device_model(self._handle.read_ranges())
        self._sectors = self._pool_sectors(self._raw)
        if self._has_points:
            self._update_points()

    def _apply_device_model(self, ranges: torch.Tensor) -> torch.Tensor:
        """Blind zone, Gaussian noise and dropout; no-returns become max_range."""
        m = self.model
        invalid = ranges < m.min_range              # blind zone → no return
        if m.range_noise_std > 0.0:
            ranges = ranges + torch.randn_like(ranges) * m.range_noise_std
        if m.dropout_prob > 0.0:
            invalid = invalid | (torch.rand_like(ranges) < m.dropout_prob)
        return torch.where(invalid,
                           torch.full_like(ranges, m.max_range),
                           ranges.clamp(0.0, m.max_range))

    def _pool_sectors(self, ranges: torch.Tensor) -> torch.Tensor:
        """[N, n_vertical, n_horizontal] → [N, n_sectors]: min over channels, then per azimuth group."""
        n_env = ranges.shape[0]
        per_sector = self.model.n_horizontal // self.n_sectors
        return ranges.min(dim=1).values.view(
            n_env, self.n_sectors, per_sector).min(dim=2).values

    def _update_points(self) -> None:
        """
        Refresh the world-frame cloud, with the same device imperfections
        applied to its validity as no-returns / dropout. Dropout is drawn
        independently from the range dropout: the cloud is a separate
        product, not a re-projection of `read_raw()`.
        """
        m = self.model
        try:
            pts, prng = self._handle.read_points()
        except (NotImplementedError, AttributeError):
            self._has_points = False
            return
        valid = (prng > m.min_range) & (prng < m.max_range)
        if m.dropout_prob > 0.0:
            valid = valid & (torch.rand_like(prng) >= m.dropout_prob)
        self._points, self._points_valid = pts, valid

    def read(self) -> torch.Tensor:
        """[N, n_sectors] sector minima of the last scan (max_range before the first)."""
        return self._sectors

    def read_raw(self) -> torch.Tensor:
        """[N, n_vertical, n_horizontal] last processed scan."""
        return self._raw

    def read_points(self):
        """World-frame hit points [N, n_beams, 3] and a validity mask
        [N, n_beams] (True = a real return, no-returns/dropouts removed)."""
        return self._points, self._points_valid

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        self._raw[envs_idx] = self.max_range
        self._sectors[envs_idx] = self.max_range
        self._points[envs_idx] = 0.0
        self._points_valid[envs_idx] = False
