"""
scripts/diagnose_bvh.py
-----------------------
Run this FIRST on one BVH file from the Zhang 2018 dataset.
It tells you exactly:
  1. All joint names in the file
  2. The coordinate convention and units
  3. The frame rate and duration

Usage:
    python scripts/diagnose_bvh.py data/dog_mocap/walk_000.bvh

Output tells you which fixes to apply in retarget_mocap.py.
"""

import sys
import numpy as np
import bvhio


def diagnose(filepath: str):
    print(f"\n{'='*60}")
    print(f"  Diagnosing: {filepath}")
    print(f"{'='*60}\n")

    root = bvhio.readAsHierarchy(filepath)

    # ── 1. Joint names ────────────────────────────────────────────
    print("ALL JOINT NAMES (in hierarchy order):")
    joints = list(root.layout())
    for joint, index, depth in joints:
        indent = "  " + "  " * depth
        print(f"{indent}[{index:02d}] {joint.Name}")

    # ── 2. Coordinate system and units ───────────────────────────
    print("\nROOT POSITION AT FRAME 0 (raw, no conversion):")
    root.loadPose(0)
    root_pos = np.array(root.PositionWorld)
    print(f"  X={root_pos[0]:.4f}  Y={root_pos[1]:.4f}  Z={root_pos[2]:.4f}")

    # bvhio is Y-up by convention. Height should be the Y component.
    height_raw = root_pos[1]
    print(f"\n  Inferred height component (Y): {height_raw:.4f}")
    if abs(height_raw) > 10:
        print(f"  → Units appear to be CENTIMETRES. Divide by 100.")
        scale_factor = 1.0 / 100.0
    elif abs(height_raw) > 0.1:
        print(f"  → Units appear to be METRES. No unit conversion needed.")
        scale_factor = 1.0
    else:
        print(f"  → Units unclear. Check manually.")
        scale_factor = 1.0

    estimated_height_m = height_raw * scale_factor
    print(f"  Estimated real-world root height: {estimated_height_m:.3f} m")
    print(f"  (German Shepherd hip height ≈ 0.55 m — does this match?)")

    # ── 3. Frame rate and duration ───────────────────────────────
    print(f"\nANIMATION METADATA:")
    # Count keyframes via the root joint
    n_frames = len(root.Keyframes)
    # bvhio stores frame time in the root keyframes
    # Estimate from a known source
    print(f"  Frames: {n_frames}")

    # Try to estimate fps from keyframe timestamps
    if n_frames >= 2:
        # bvhio Keyframe objects have a 'Time' attribute
        try:
            t0 = root.Keyframes[0].Time
            t1 = root.Keyframes[1].Time
            dt = t1 - t0
            fps = 1.0 / dt if dt > 0 else 0
            print(f"  Frame time: {dt:.6f}s")
            print(f"  Frame rate: {fps:.1f} Hz")
            print(f"  Duration:   {n_frames * dt:.2f}s")
        except AttributeError:
            print(f"  (Could not read frame time — assume 30 fps)")

    # ── 4. Identify leaf joints (likely feet) ─────────────────────
    print(f"\nLEAF JOINTS (likely feet/end effectors):")
    for joint, index, depth in joints:
        if not list(joint.Children):   # has no children
            pos = np.array(joint.PositionWorld)
            print(f"  [{index:02d}] {joint.Name:30s}  "
                  f"pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

    # ── 5. Summary ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY — what to set in retarget_mocap.py:")
    print(f"{'='*60}")
    print(f"  UNIT_SCALE     = {scale_factor}   "
          f"{'(cm→m)' if scale_factor == 0.01 else '(already metres)'}")
    print(f"  Convention:      Y-up (bvhio always uses Y-up)")
    print(f"  Conversion:      Y→Z swap required for Genesis (Z-up)")
    print(f"\n  → Look at the joint names above.")
    print(f"    Identify which 4 are the feet (lowest leaf joints)")
    print(f"    and fill in FOOT_JOINT_NAMES in retarget_mocap.py.\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/diagnose_bvh.py <path_to_bvh_file>")
        sys.exit(1)
    diagnose(sys.argv[1])
