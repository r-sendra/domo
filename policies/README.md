# Stable policies

The **blessed** checkpoints the skills point at, resolved by symbolic name
through `domo.policies` (not by brittle `runs/.../checkpoint_final_*.pt` paths).

| name    | file       | promoted from                              | role                     |
|---------|------------|--------------------------------------------|--------------------------|
| `walk`  | `walk.pt`  | `runs/go2_cpg/checkpoint_final_coupled.pt` | most stable CPG gait     |
| `avoid` | `avoid.pt` | `runs/go2_cpg/checkpoint_final_avoid.pt`   | lidar-avoidance net (Δv) |

## Usage

```python
from domo.policies import stable_policy, load_stable_locomotion, stable_go2_library

path        = stable_policy("walk")             # absolute path to walk.pt
walk_fn, _  = load_stable_locomotion(device)    # ready-to-use policy callable
library     = stable_go2_library(lidar, device) # walk (+avoid when lidar given)
```

## Promoting a new checkpoint

Copy it over the corresponding file here, e.g.:

```
cp runs/go2_cpg/checkpoint_final_newbest.pt policies/walk.pt
```

Register additional skills by adding to `STABLE` in `domo/policies.py`.

> The `*.pt` files are git-ignored (like `runs/`); this is a curated local
> cache. The registry (`domo/policies.py`) and this README are versioned.
