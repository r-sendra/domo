"""
scripts/validate_retargeting.py
--------------------------------
Run after retarget_mocap.py to verify the reference dataset is correct
before using it for AMP training.

Checks:
  1. Basic statistics — no NaN, no extreme values
  2. Foot height oscillation — should alternate between ground (~0) and air
  3. Base height — should be near Go2 nominal height (0.34m)
  4. Projected gravity — Z component should be near -1.0 (upright)
  5. Gait symmetry — FL/FR and RL/RR feet should alternate

Usage:
    python scripts/validate_retargeting.py
    python scripts/validate_retargeting.py data/amp_go2_reference.npy
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")   # no display needed
import matplotlib.pyplot as plt
from pathlib import Path


# State layout (must match retarget_mocap.py)
IDX = {
    "FL_pos": slice(0,  3),
    "FR_pos": slice(3,  6),
    "RL_pos": slice(6,  9),
    "RR_pos": slice(9, 12),
    "proj_gravity": slice(12, 15),
    "base_height":  slice(15, 16),
    "lin_vel":      slice(16, 19),
    "ang_vel":      slice(19, 22),
}

# Go2 nominal values
GO2_NOMINAL_HEIGHT  = 0.34   # metres
GO2_MAX_FOOT_HEIGHT = 0.30   # maximum reasonable foot lift


def check_statistics(data: np.ndarray) -> bool:
    """Basic sanity checks on the dataset."""
    ok = True
    print("1. BASIC STATISTICS")

    if np.any(np.isnan(data)):
        print(f"   ❌ NaN values found: {np.sum(np.isnan(data))} elements")
        ok = False
    else:
        print(f"   ✅ No NaN values")

    if np.any(np.isinf(data)):
        print(f"   ❌ Inf values found: {np.sum(np.isinf(data))} elements")
        ok = False
    else:
        print(f"   ✅ No Inf values")

    print(f"   Min:  {data.min():.4f}")
    print(f"   Max:  {data.max():.4f}")
    print(f"   Mean: {data.mean():.4f}")
    print(f"   Std:  {data.std():.4f}")

    if data.max() > 10.0:
        print(f"   ❌ Max value {data.max():.2f} > 10 — likely unit error (forgot /100?)")
        ok = False
    if data.min() < -10.0:
        print(f"   ❌ Min value {data.min():.2f} < -10 — likely unit error")
        ok = False

    return ok


def check_base_height(data: np.ndarray) -> bool:
    """Base height should be near Go2 nominal standing height."""
    ok = True
    print("\n2. BASE HEIGHT")

    heights = data[:, IDX["base_height"]].flatten()
    mean_h  = heights.mean()
    std_h   = heights.std()

    print(f"   Mean: {mean_h:.3f} m  (expected ~{GO2_NOMINAL_HEIGHT} m)")
    print(f"   Std:  {std_h:.3f} m")
    print(f"   Min:  {heights.min():.3f} m")
    print(f"   Max:  {heights.max():.3f} m")

    if abs(mean_h - GO2_NOMINAL_HEIGHT) > 0.20:
        print(f"   ❌ Mean height {mean_h:.3f} far from Go2 nominal {GO2_NOMINAL_HEIGHT} m")
        print(f"      → Check UNIT_SCALE and MORPHOLOGY_SCALE in retarget_mocap.py")
        ok = False
    else:
        print(f"   ✅ Base height looks correct")

    return ok


def check_projected_gravity(data: np.ndarray) -> bool:
    """Projected gravity Z should be near -1.0 when robot is upright."""
    ok = True
    print("\n3. PROJECTED GRAVITY")

    grav = data[:, IDX["proj_gravity"]]
    gz   = grav[:, 2]

    print(f"   Gravity X mean: {grav[:,0].mean():.3f}  (expected ~0)")
    print(f"   Gravity Y mean: {grav[:,1].mean():.3f}  (expected ~0)")
    print(f"   Gravity Z mean: {gz.mean():.3f}  (expected ~-1)")

    if abs(gz.mean() - (-1.0)) > 0.3:
        print(f"   ❌ Gravity Z mean {gz.mean():.3f} far from -1.0")
        print(f"      → Coordinate conversion may be wrong")
        ok = False
    else:
        print(f"   ✅ Projected gravity looks correct")

    return ok


def check_foot_heights(data: np.ndarray) -> bool:
    """Foot Z positions should oscillate — ground contact and swing phase."""
    ok = True
    print("\n4. FOOT HEIGHTS (Z component of foot positions in base frame)")

    feet = {
        "FL": data[:, IDX["FL_pos"]][:, 2],
        "FR": data[:, IDX["FR_pos"]][:, 2],
        "RL": data[:, IDX["RL_pos"]][:, 2],
        "RR": data[:, IDX["RR_pos"]][:, 2],
    }

    for leg, fz in feet.items():
        min_h = fz.min()
        max_h = fz.max()
        std_h = fz.std()
        print(f"   {leg}: min={min_h:.3f}  max={max_h:.3f}  std={std_h:.3f}")

        if std_h < 0.005:
            print(f"      ❌ {leg} foot barely moves — no gait detected")
            ok = False
        if max_h > GO2_MAX_FOOT_HEIGHT:
            print(f"      ⚠️  {leg} foot lifts to {max_h:.3f} m — possible scale issue")

    if ok:
        print(f"   ✅ Foot heights show gait oscillation")

    return ok


def check_gait_symmetry(data: np.ndarray) -> bool:
    """FL and FR feet should alternate (trot) or be in-phase (pace)."""
    print("\n5. GAIT SYMMETRY")

    fl_z = data[:1000, IDX["FL_pos"]][:, 2]
    fr_z = data[:1000, IDX["FR_pos"]][:, 2]

    correlation = np.corrcoef(fl_z, fr_z)[0, 1]
    print(f"   FL-FR foot height correlation: {correlation:.3f}")

    if abs(correlation) > 0.3:
        if correlation < 0:
            print(f"   ✅ Trot-like gait detected (FL and FR alternate, corr={correlation:.2f})")
        else:
            print(f"   ✅ Pace-like gait detected (FL and FR in phase, corr={correlation:.2f})")
    else:
        print(f"   ⚠️  Low correlation — irregular or mixed gaits")

    return True


def plot_diagnostics(data: np.ndarray, output_dir: str = "data/"):
    """Save diagnostic plots to disk."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    n_plot = min(300, len(data))   # plot first 300 frames (10s at 30fps)
    t      = np.arange(n_plot) / 30.0   # time in seconds

    # Plot 1: Foot heights
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    for ax, (leg, col) in zip(axes, [
        ("FL", 2), ("FR", 5), ("RL", 8), ("RR", 11)
    ]):
        fz = data[:n_plot, col]
        ax.plot(t, fz, linewidth=1)
        ax.axhline(0, color='r', linestyle='--', alpha=0.5, label="ground")
        ax.set_ylabel(f"{leg} foot Z (m)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title("Foot Heights in Base Frame (should oscillate with gait)")
    plt.tight_layout()
    path = f"{output_dir}/validate_feet.png"
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"\n   Saved: {path}")

    # Plot 2: Base height and gravity
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    heights = data[:n_plot, 15]
    axes[0].plot(t, heights, linewidth=1)
    axes[0].axhline(GO2_NOMINAL_HEIGHT, color='r', linestyle='--',
                    label=f"Go2 nominal ({GO2_NOMINAL_HEIGHT}m)")
    axes[0].set_ylabel("Base height (m)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Base Height (should be near 0.34m)")

    gz = data[:n_plot, 14]   # gravity Z
    axes[1].plot(t, gz, linewidth=1)
    axes[1].axhline(-1.0, color='r', linestyle='--', label="upright (-1.0)")
    axes[1].set_ylabel("Gravity Z (base frame)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Projected Gravity Z (should be near -1.0 when upright)")
    plt.tight_layout()
    path = f"{output_dir}/validate_base.png"
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"   Saved: {path}")


def validate(dataset_path: str):
    print(f"\n{'='*55}")
    print(f"  Validating: {dataset_path}")
    print(f"{'='*55}\n")

    data = np.load(dataset_path)
    print(f"Shape: {data.shape}")
    print(f"Frames: {len(data)}  ({len(data)/30:.1f}s at 30fps)\n")

    results = [
        check_statistics(data),
        check_base_height(data),
        check_projected_gravity(data),
        check_foot_heights(data),
        check_gait_symmetry(data),
    ]

    print("\n6. DIAGNOSTIC PLOTS")
    output_dir = str(Path(dataset_path).parent)
    plot_diagnostics(data, output_dir)

    print(f"\n{'='*55}")
    if all(results):
        print(f"  ✅ ALL CHECKS PASSED — dataset ready for AMP training")
    else:
        n_failed = sum(1 for r in results if not r)
        print(f"  ❌ {n_failed} CHECK(S) FAILED — fix issues before training")
        print(f"     Review the output above and adjust retarget_mocap.py")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/amp_go2_reference.npy"
    validate(path)
