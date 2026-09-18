# DOMO documentation

Start here if you are new; the pages are meant to be read in this order.

| Page | What it covers |
|------|----------------|
| [getting-started.md](getting-started.md) | installing the environment, verifying it, first run on CPU, first training run on GPU |
| [project-overview.md](project-overview.md) | what DOMO is, the M1–M8 roadmap, the twin-first principle, repository layout, vocabulary |
| [architecture.md](architecture.md) | the layer stack, dependency rules, the two runtimes (twin and learning), control hierarchy, sim ↔ real map |
| [conventions.md](conventions.md) | quaternions, joint order, tensors, layering, naming, code style |
| [running.md](running.md) | training, evaluation, checkpoints, the stable-policy registry, the live dashboard |
| [examples.md](examples.md) | every example: purpose, commands, flags, expected output |
| [extending.md](extending.md) | how to add a task, a reward term, a skill and card, a sensor, a scene, a physics backend, an LLM provider, an Eureka task spec |
| [troubleshooting.md](troubleshooting.md) | Genesis quirks, checkpoint issues, LLM providers, dashboard, tests |

## API reference

One page per package, generated from the source and kept in sync by hand.

| Package | Page |
|---------|------|
| `domo.sim` | [api/sim.md](api/sim.md) |
| `domo.robot` | [api/robot.md](api/robot.md) |
| `domo.control` | [api/control.md](api/control.md) |
| `domo.skills` | [api/skills.md](api/skills.md) |
| `domo.tasks` | [api/tasks.md](api/tasks.md) |
| `domo.rl` | [api/rl.md](api/rl.md) |
| `domo.eureka` | [api/eureka.md](api/eureka.md) |
| `domo.llm`, `domo.hri` | [api/llm-hri.md](api/llm-hri.md) |
| `domo.world`, `domo.scenes`, `domo.policies`, `domo.checkpoints`, `domo.utils` | [api/world-and-services.md](api/world-and-services.md) |
| `domo.dashboard` | [api/dashboard.md](api/dashboard.md) |
