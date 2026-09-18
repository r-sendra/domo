# `domo.skills` — cards, grammar, compiler, programs

`domo.skills` is the skill library layer (M5). It gives every primitive in
`domo.control` an LLM-legible card, defines a small composition grammar
(`@` layering, `>>` sequence, `|` fallback, `.for/.until/.repeat`
modifiers), type-checks programs against the cards, and runs the result as
a `CompositeSkill`, an ordinary skill that any `Controller` can host.
`PlanningController` closes the loop: programs are authored at runtime,
inside the controller, by whatever intelligence occupies `plan()`.

It sits above `domo.control` (the primitives it composes) and below
`domo.world` and `examples/`. Pure torch, engine-free: it can be imported
and tested on a machine without Genesis (`tests/test_skill_grammar.py`,
`tests/test_skills_library.py`, `tests/test_planner.py` run on fakes).

```python
from domo.skills import make_go2_library

library = make_go2_library(walk_policy, avoid_policy, lidar)
program = library.compile("(avoid @ walk(vx=0.6)).until(moved(3)) >> stand.for(2)",
                          device=robot.device)
program.setup(robot)                 # a CompositeSkill is an ordinary Skill
print(library.describe())            # the catalogue an LLM plans against
```

## Module map

| Module | Public names | Role |
|--------|--------------|------|
| `card.py` | `MOTOR`, `CMD_VELOCITY`, `ParamSpec`, `SkillCard` | structured description of one skill; `describe()` renders the prompt block |
| `conditions.py` | `Condition`, `ConditionRegistry`, `standard_conditions` | named predicates `cond(state, t_s) -> bool [N]` |
| `grammar.py` | `parse`, `parse_condition`, `GrammarError`, AST classes (`SkillRef`, `Layer`, `Sequence`, `Fallback`, `Modified`, `CondRef`) | tokenizer + recursive-descent parser, canonical `to_text()` |
| `library.py` | `SkillLibrary`, `CompileError` | registry of cards, factories and conditions; the compiler |
| `nodes.py` | `RUNNING`, `SUCCESS`, `FAILURE`, `CompositeSkill`, `ExecNode` subclasses | the executable tree and its status/trace protocol |
| `planner.py` | `PlanningController`, `PlanOutcome` | a `Controller` whose programs come from `plan()` |
| `builtin.py` | `STAND_CARD`, `WALK_CARD`, `AVOID_CARD`, `FORWARD_CARD`, `BACKWARD_CARD`, `GOTO_CARD`, `SLAM_CARD`, `make_go2_library` | the current Go2 repertoire |

Everything above except the AST classes and `ExecNode` subclasses is
re-exported from `domo.skills`; import the rest from `domo.skills.grammar`
and `domo.skills.nodes`.

## Cards

### `MOTOR`, `CMD_VELOCITY`

The two interface strings, the grammar's type system:

| Constant | Value | Meaning |
|----------|-------|---------|
| `MOTOR` | `"motor"` | emits joint targets; can stand at an execution position and terminate a composition |
| `CMD_VELOCITY` | `"command:velocity"` | emits `(Δvx, Δvy, Δvyaw)`; must be layered with `@` on a motor skill whose `accepts == "velocity"` |

### `ParamSpec` (frozen dataclass)

One grammar parameter of a skill (`skill(name=value)`). Values are floats
in the grammar; `range` is enforced at compile time.

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `name` | `str` | — | parameter name in program text |
| `description` | `str` | — | one line for the prompt |
| `default` | `float` | `0.0` | used when the program omits the parameter |
| `range` | `tuple[float, float] \| None` | `None` | inclusive bounds; `None` = unchecked |
| `unit` | `str` | `""` | shown in `describe()` |

### `SkillCard` (dataclass)

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `name` | `str` | — | the skill's grammar name |
| `description` | `str` | — | one natural-language paragraph |
| `interface` | `str` | `MOTOR` | `MOTOR` or `CMD_VELOCITY` |
| `accepts` | `str \| None` | `None` | command channel a motor skill consumes (`"velocity"`) |
| `params` | `list[ParamSpec]` | `[]` | motor skills list the channel components (`vx, vy, vyaw`); command skills list their goal parameters |
| `preconditions` | `list[str]` | `[]` | natural language, not machine-checked |
| `effects` | `list[str]` | `[]` | natural language, not machine-checked |
| `success_when` | `list[str]` | `[]` | condition expressions; `[]` = runs until a modifier ends it |
| `fail_when` | `list[str]` | `[]` | condition expressions; any env true → `FAILURE` |
| `constraints` | `dict[str, tuple[float, float]]` | `{}` | hard clamp on the consumed channel, applied after layering |
| `safety_notes` | `list[str]` | `[]` | documentation |
| `max_duration_s` | `float \| None` | `None` | failsafe: `FAILURE` after this long in the node |

`success_when` and `fail_when` of a **motor** card are compiled with
`parse_condition` and must be valid condition syntax (`tipped(0.9)`). On a
**command** card only `fail_when` is compiled; its `success_when` is
descriptive text for the prompt (completion comes from the skill's
`success_flags`).

| Member | Returns |
|--------|---------|
| `channel` | `"velocity"` for a command card (`interface.split(":")[1]`), `None` for motor |
| `param(name)` | the `ParamSpec`; `KeyError("skill 'x' has no parameter 'y'")` otherwise |
| `describe()` | the prompt block below |

`WALK_CARD.describe()` renders as:

```
SKILL walk  [motor, accepts 'velocity']
  Omnidirectional velocity-tracking locomotion (learned CPG policy). Tracks a body-frame velocity command; ...
  parameters:
    - vx=0.0 [m/s] range=[-1.0, 2.0]: forward velocity
    - vy=0.0 [m/s] range=[-0.5, 0.5]: lateral velocity (positive = left)
    - vyaw=0.0 [rad/s] range=[-1.5, 1.5]: yaw rate (positive = turn left)
  preconditions: standing on ground; trained checkpoint loaded
  effects: robot moves at the commanded velocity
  succeeds when: (never by itself — bound its duration with .for(seconds) or .until(condition))
  fails when: tipped(0.9); fallen(0.18)
  command constraints: vx∈[-1.0,2.0], vy∈[-0.5,0.5], vyaw∈[-1.5,1.5]
  safety: blind to obstacles; commands clamped to constraints
```

A command card's header reads `[command:velocity, drives the 'velocity'
channel of a base skill]`; `max_duration_s` adds `failsafe: aborts after
Ns`.

## Conditions

### `Condition`

```python
Condition = Callable[[object, float], torch.Tensor]   # cond(state, t_s) -> bool [N]
```

`t_s` is the number of seconds since the enclosing node was entered (the
node-local clock), not program time. Conditions get no `reset()`; a
stateful one must detect re-entry itself (see `moved`).

### `ConditionRegistry`

| Method | Behaviour |
|--------|-----------|
| `register(name, factory)` | `factory(*args) -> Condition`; replaces an existing name |
| `names()` | sorted list, printed in `library.describe()` |
| `make(name, *args)` | instantiate from grammar literals; `KeyError("unknown condition 'x' (known: [...])")` |

### `standard_conditions() -> ConditionRegistry`

The state-only conditions. Sensor-bound ones are closures registered by the
caller (`make_go2_library` adds `blocked`/`clear` when it has a lidar).

| Condition | True when | Default |
|-----------|-----------|---------|
| `timeout(T)` | `t_s >= T` (same value for every env) | — |
| `tipped(rad)` | `max(|roll|, |pitch|) > rad` (`base_euler[:, :2]`) | 0.7 |
| `fallen(h)` | `base_pos[:, 2] < h` | 0.18 |
| `still(v)` | `‖base_lin_vel‖ < v` | 0.05 |
| `moved(d)` | planar displacement from node entry `> d` | — |
| `blocked(d)` | `min(lidar.read()) < d` (library with lidar only) | 0.25 |
| `clear(d)` | `min(lidar.read()) > d` (library with lidar only) | 1.4 |

`moved` latches its origin on the first evaluation and re-latches whenever
`t_s` goes backwards (the enclosing node was re-entered); the heuristic
fails only if a node is re-entered and evaluated at a `t_s` not smaller
than the last one it saw.

## Grammar

### Syntax

```
program  := seq
seq      := fallback ('>>' fallback)*          # S1 then S2 (when S1 succeeds)
fallback := layer ('|' layer)*                 # S1, or S2 if S1 fails
layer    := post ('@' post)*                   # S1 executed ON TOP OF S2
                                               #   right-assoc: the rightmost
                                               #   element is the motor skill
post     := atom modifier*
modifier := '.for(' NUM ')'                    # succeed after NUM seconds
          | '.until(' cond ')'                 # succeed when cond holds (all envs)
          | '.repeat(' NUM ')'                 # loop the child int(NUM) times
atom     := NAME [ '(' kv (',' kv)* ')' ]      # skill with parameters
          | '(' seq ')'
kv       := NAME '=' NUM
cond     := NAME [ '(' NUM (',' NUM)* ')' ]

NAME     := [A-Za-z_][A-Za-z0-9_]*
NUM      := -?\d+\.?\d*                        # 2, -0.5, 3. — but not .5 or 1e3
```

Whitespace is insignificant. Tokens are `>>`, `@`, `|`, `.`, `(`, `)`,
`,`, `=`, `NUM`, `NAME`.

### Precedence

| Level | Operator | Binds |
|-------|----------|-------|
| tightest | `.for(...)`, `.until(...)`, `.repeat(...)` | to the atom or parenthesised group on their left; chained modifiers nest, `x.for(2).repeat(3)` = `Modified(repeat=3, Modified(for=2, x))` (the repeat loops the timed node) |
| | `@` | right-associative: `a @ b @ walk` = `Layer(a, Layer(b, walk))` |
| | `\|` | left to right |
| loosest | `>>` | left to right |

So `avoid @ walk(vx=0.6) | stand >> walk.for(3)` parses as
`Sequence(Fallback(Layer(avoid, walk), stand), Modified(walk, for=3))`, and
`(avoid @ walk).for(4)` needs its parentheses while `avoid @ walk.for(4)`
would attach `.for` to `walk` (and then fail to compile, see below).

### Valid programs

```
walk(vx=0.5).for(4)
stand.for(2)
(avoid @ walk(vx=0.6)).until(moved(3)) >> stand.for(2)
walk(vx=0.5).for(5) | stand.for(1)
(walk(vx=0.5).for(2)).repeat(3)
goto(x=2, y=1) @ walk >> goto(x=0, y=0) @ walk
forward(distance=2, speed=0.4) @ walk
(slam @ avoid @ goto(x=8, y=5) @ walk).until(timeout(60))
((avoid @ walk(vx=0.6)).for(6) >> walk(vyaw=0.8).for(2)).repeat(4)
```

### `parse(text)` and the AST

`parse(text)` returns the root node; every node has `to_text()` printing
the canonical form with minimal parentheses, and
`parse(node.to_text()).to_text() == node.to_text()`.

| Class | Fields | `to_text()` |
|-------|--------|-------------|
| `SkillRef` | `name`, `params: dict[str, float]` | `walk(vx=0.5)`; integral floats print without `.0` |
| `Layer` | `top`, `base` | `top @ base` |
| `Sequence` | `children` | `a >> b`, children parenthesised if they are sequences |
| `Fallback` | `children` | `a \| b`, children parenthesised if sequences or fallbacks |
| `Modified` | `child`, `for_s`, `until: CondRef`, `repeat` | child parenthesised if sequence, fallback or layer, then `.for(T).until(c).repeat(n)` |
| `CondRef` | `name`, `args: tuple[float, ...]` | `tipped(0.9)` |

`((avoid @ walk) | stand).for(2) >> (walk(vx=1).for(1)).repeat(2)`
canonicalises to
`(avoid @ walk | stand).for(2) >> walk(vx=1).for(1).repeat(2)`.
`CompositeSkill.source` and `PlanOutcome.program` hold this canonical text.

`parse_condition(text) -> CondRef` parses a lone condition expression
(card fields).

### `GrammarError`

A `ValueError` for malformed text. Messages:

| Input | Message |
|-------|---------|
| `walk $ stand` | `unexpected character '$' at 5 in: walk $ stand` |
| `walk >>` | `expected NAME, got EOF '' in: walk >>` |
| `walk(vx=)` | `expected NUM, got RP ')' in: walk(vx=)` |
| `walk(vx=.5)` | `expected NUM, got DOT '.' in: walk(vx=.5)` |
| `walk stand` | `trailing input from token 'stand' in: walk stand` |
| `walk.sometimes(3)` | `unknown modifier '.sometimes(' (known: for, until, repeat)` |
| `parse_condition("still >> x")` | `trailing input in condition: still >> x` |

## `SkillLibrary`

```python
SkillLibrary(conditions: ConditionRegistry | None = None)   # default: standard_conditions()
```

| Method | Behaviour |
|--------|-----------|
| `register(card, factory)` | `factory() -> Skill \| CommandSkill`, a fresh instance per call; replaces an existing name |
| `register_condition(name, factory)` | adds to `library.conditions` |
| `card(name)` | the `SkillCard`; `CompileError("unknown skill ...")` otherwise |
| `names()` | sorted skill names |
| `describe()` | every card's `describe()` block plus a `COMPOSITION GRAMMAR` cheat-sheet listing the operators, modifiers, registered condition names and precedence; the planning prompt |
| `compile(program_text, device="cpu")` | parse, type-check, instantiate → `CompositeSkill`. `device` is where constant command tensors live (pass `robot.device`) |

### `CompileError`

A `ValueError` for a well-formed program that violates the type rules. The
checks, in the order the compiler applies them, with the library built by
`make_go2_library(walk, avoid, lidar)`:

| Rule | Example | Message |
|------|---------|---------|
| skill must exist | `fly.for(1)` | `unknown skill 'fly' (library has: ['avoid', 'backward', 'forward', 'goto', 'slam', 'stand', 'walk'])` |
| a command skill cannot stand alone | `avoid.for(3)` | `'avoid' is a command:velocity skill — it cannot run alone; layer it: 'avoid @ <motor skill>'` |
| left of `@` must be a single skill reference | `avoid.for(2) @ walk`, `(avoid \| slam) @ walk` | `the left side of '@' must be a single command skill` |
| left of `@` must be a command skill | `stand @ walk` | `'stand' is a motor skill — only command skills can be layered on top ('@'). Did you mean 'walk >> stand'?` |
| parameters must exist on the card | `walk(speed=1).for(1)`, `avoid(k=1) @ walk` | `skill 'walk' has no parameter 'speed'` |
| parameters must be in range | `walk(vx=5).for(1)` | `walk.vx=5.0 outside allowed range [-1.0, 2.0]` |
| a motor skill without a channel takes no parameters | `stand(vx=1).for(1)` | `'stand' takes no parameters (got {'vx': 1.0})` |
| right of `@` must be a motor leaf or another layer | `avoid @ (walk(vx=0.5).for(3))`, `avoid @ (walk >> stand)` | `the right side of '@' must be a (possibly layered) motor skill — apply modifiers to the whole layer instead: '(avoid @ walk(vx=0.5).for(3)).for(...)'` |
| channels must match down to the motor leaf | `avoid @ stand` | `cannot layer 'avoid' (drives 'velocity') on 'stand' (accepts 'nothing')` |
| the factory must return a `CommandSkill` | a mis-registered card | `'x' card says command:velocity but the instance is not a CommandSkill` |

Unknown conditions are different: `walk.until(flying)` and a card whose
`fail_when` names an unregistered condition raise the registry's
`KeyError("unknown condition 'flying' (known: [...])")`, not a
`CompileError` (see Gotchas).

### Instance policy

* **Motor skills are instantiated once per program and shared** by every
  reference to the same name, keyed by name in `CompositeSkill.instances`
  (`"walk"`). Their per-leaf parameters are constant command vectors owned
  by the `MotorLeaf`, not by the skill.
* **Each `@` occurrence gets a fresh command-skill instance**, keyed
  `name#k` (`"goto#1"`, `"goto#2"`), because command skills carry
  per-occurrence goal state; the grammar parameters are applied with
  `configure(**params)` at compile time.

```python
prog = library.compile("goto(x=2, y=1) @ walk >> goto(x=0, y=0) @ walk")
sorted(prog.instances)                  # ['goto#1', 'goto#2', 'walk']
prog.instances["goto#1"].target         # (2.0, 1.0)
```

The velocity constant of a motor leaf is `[vx, vy, vyaw]` from the card
defaults overridden by the program's parameters; the clamp bounds come from
`card.constraints`, `±1e9` for an unconstrained component.

## Execution

### Status

```python
RUNNING, SUCCESS, FAILURE = 0, 1, 2
```

Every node exposes `.status` after each update. Two global rules:

* **Transitions are global across envs.** `SUCCESS` needs every env
  (`flags.all()`), `FAILURE` triggers on any env (`flags.any()`).
  Compositions are meant for deployment (`n_envs = 1`); training is the
  job of tasks.
* **Containers act on a child's status at the start of their next tick.**
  The tick on which a child reaches `SUCCESS` still returns that child's
  output (its targets, or the hold pose if it was a modifier); the next
  tick enters and runs the successor. `FAILURE` is additionally
  propagated up through `SequenceNode` and `ModifiedNode` on the same tick
  (a `FallbackNode` ancestor still intercepts it on its next tick).

Finished nodes return `ctx.hold()`, the default stance `[N, D]`.

### Node semantics

**`MotorLeaf`** (a motor skill at an execution position). Each tick:
`command = clamp(const_cmd + extra_cmd, lo, hi)` is written into
`skill.command` (only for skills with a command channel), then
`skill.update(state, dt)`. Then, in priority order:

1. any `fail_when` condition true on any env → `FAILURE`;
2. `t_local >= card.max_duration_s` → `FAILURE`;
3. all `success_when` conditions... any of them true on every env →
   `SUCCESS` (conditions in a list are OR-ed, then `all()` over envs);
4. otherwise `RUNNING`. An empty `success_when` runs until a modifier or
   parent ends it.

`enter()` resets the motor skill for all envs (oscillators restart on every
leg that uses `walk`).

**`LayerNode`** (`top @ base`). Each tick `delta = top.update_command(state, dt)`.
Additive top: `total = delta + extra_cmd`. Override top
(`additive=False`): `delta -= motor_leaf.const_cmd`, so after the leaf adds
its constant back the effective command equals the override skill's output
plus any layers stacked above it. `base.update(state, dt, extra_cmd=total)`.
Status mirrors the base, then `SUCCESS` when `top.success_flags(state)` is
all-true (logged `goal reached`), then `FAILURE` when any of the top card's
`fail_when` conditions holds on any env. `enter()` resets the top skill for
all envs, then enters the base.

Layering maths with the built-in cards:

```
avoid @ walk(vx=0.6)               walk.command = clamp([0.6,0,0] + Δavoid, walk.constraints)
goto(x=1,y=0) @ walk(vx=0.9)       walk.command = clamp(goto_cmd, ...)          (vx=0.9 ignored)
avoid @ goto(x=1,y=0) @ walk       walk.command = clamp(goto_cmd + Δavoid, ...)
slam @ avoid @ walk(vx=0.6)        walk.command = clamp([0.6,0,0] + Δavoid + 0, ...)
```

**`SequenceNode`** (`a >> b`). On a tick where the current child is
`FAILURE` → `FAILURE`, hold. Where it is `SUCCESS` → enter the next child
(log `sequence → step i/n`) or, after the last, `SUCCESS`, hold. Otherwise
run the child; if it fails during this tick, the sequence fails now.

**`FallbackNode`** (`a | b`). Current child `SUCCESS` → `SUCCESS`, hold.
`FAILURE` → enter the next child (log `fallback → alternative i/n`) or,
after the last, `FAILURE`, hold. Otherwise run the child (a same-tick child
failure is passed through as targets first).

**`ModifiedNode`** (`.for/.until/.repeat`). Own criteria first, on the fresh
state: `.for(T)` with `t_local >= T` → `SUCCESS`, hold; `.until(cond)` with
`cond(state, t_local).all()` → `SUCCESS`, hold. Then child `FAILURE` →
`FAILURE`, hold. Child `SUCCESS` → count a run; with `.repeat(n)` and runs
remaining, re-enter the child (log `repeat k/n`) and run it this tick,
else `SUCCESS`, hold. Otherwise run the child; a same-tick child failure
fails the node.

### `CompositeSkill`

```python
CompositeSkill(root, ctx, instances, source)    # built by SkillLibrary.compile
```

Duck-typed as a `Skill` (not a subclass): `setup`, `reset_idx`, `update`,
`name = "program[<source>]"`.

| Member | Behaviour |
|--------|-----------|
| `setup(robot)` | `setup` on every instance, allocate the hold pose |
| `reset_idx(envs_idx)` | `reset_idx` on every instance for `envs_idx`, then **restart the whole program**: `t = 0`, trace cleared, `start: <source>` logged, root re-entered. Program counters are global, so a partial reset restarts everything |
| `update(state, dt)` | one tick of the tree, then `t += dt`; once finished returns the hold pose without running anything |
| `status` | `RUNNING \| SUCCESS \| FAILURE` of the root |
| `finished`, `succeeded` | `status != RUNNING`, `status == SUCCESS` |
| `trace` | a copy of the timestamped transition log |
| `source` | canonical program text |
| `instances` | `{"walk": ..., "goto#1": ...}` |

### Worked timeline

`walk(vx=0.5).for(0.04) >> stand.for(0.04)` at `dt = 0.02`, after
`reset_idx` (which logs `start` and enters `walk`):

| Tick | `t` at start | What happens | Output |
|------|--------------|--------------|--------|
| 1 | 0.00 | `.for`: `t_local = 0 < 0.04`; `walk` runs | walk targets |
| 2 | 0.02 | `t_local = 0.02`; `walk` runs | walk targets |
| 3 | 0.04 | `.for` fires → modifier `SUCCESS`, logs `walk → SUCCESS (.for 0.04s)`; the sequence only sees this next tick | hold |
| 4 | 0.06 | sequence advances: `sequence → step 2/2 (stand.for(0.04))`, `enter stand`; `t_local = 0`; `stand` runs | stand targets |
| 5 | 0.08 | `t_local = 0.02`; `stand` runs | stand targets |
| 6 | 0.10 | `.for` fires → `stand → SUCCESS (.for 0.04s)` | hold |
| 7 | 0.12 | sequence sees `SUCCESS`, no more children → `SUCCESS`; `CompositeSkill` logs `program → SUCCESS` after advancing `t` | hold |
| 8+ | | `finished`; `update` returns hold without touching the tree | hold |

The resulting trace:

```
[t=   0.00s] start: walk(vx=0.5).for(0.04) >> stand.for(0.04)
[t=   0.00s] enter walk
[t=   0.04s] walk → SUCCESS (.for 0.04s)
[t=   0.06s] sequence → step 2/2 (stand.for(0.04))
[t=   0.06s] enter stand
[t=   0.10s] stand → SUCCESS (.for 0.04s)
[t=   0.14s] program → SUCCESS
```

Failure is faster. `walk(vx=0.5).for(5) | stand.for(0.1)` with the robot
pitched past `tipped(0.9)`: tick 1 runs `walk`, the leaf logs
`walk → FAILURE (safety/abort condition)` and the modifier fails the same
tick; tick 2 the fallback logs `fallback → alternative 2/2 (stand.for(0.1))`
and runs `stand`. A goal-directed layer works like a modifier:
`goto(x=1, y=0) @ walk >> stand.for(1)` with the robot already within
`tol_pos` logs `goto → SUCCESS (goal reached)` on tick 1 (the leaf still
runs with `goto`'s zero command) and `sequence → step 2/2` on tick 2.

### Trace format

Every line is `[t=<program time, 7.2f>s] <message>`. Trace lines are part
of the contract: tests and planners grep them.

| Message | Emitted by |
|---------|------------|
| `start: <source>` | `CompositeSkill.reset_idx` |
| `enter <name>` | `MotorLeaf.enter` |
| `layer <top> active` | `LayerNode.enter` |
| `<name> → SUCCESS` | motor leaf, card `success_when` |
| `<name> → FAILURE (safety/abort condition)` | motor leaf, card `fail_when` |
| `<name> → FAILURE (failsafe max_duration <T>s)` | motor leaf, `max_duration_s` |
| `<top> → SUCCESS (goal reached)` | layer, `success_flags` all true |
| `<top> → FAILURE (layer abort condition)` | layer, top card `fail_when` |
| `sequence → step <i>/<n> (<label>)` | `SequenceNode` |
| `fallback → alternative <i>/<n> (<label>)` | `FallbackNode` |
| `<label> → SUCCESS (.for <T>s)` | `ModifiedNode` |
| `<label> → SUCCESS (.until <cond>)` | `ModifiedNode` |
| `repeat <k>/<n> (<label>)` | `ModifiedNode`, before re-entering |
| `program → SUCCESS` / `program → FAILURE` | `CompositeSkill.update` |

Labels are built from node labels: a leaf is its name, a layer is
`top @ base`, sequences and fallbacks join with ` >> ` and ` | `, and a
modifier appends `.for(T)`, `.until(text)`, `.repeat(n)`. Labels carry no
parentheses.

## Planning

### `PlanOutcome` (dataclass)

| Field | Type | Meaning |
|-------|------|---------|
| `program` | `str` | the text `plan()` returned (canonical form when it ran) |
| `succeeded` | `bool` | `CompositeSkill.succeeded`; `False` for a compile error |
| `trace` | `list[str]` | the program's trace; empty for a compile error |
| `compile_error` | `str \| None` | the `CompileError`/`GrammarError` message when the program never ran |

### `PlanningController`

```python
PlanningController(library: SkillLibrary, decision_interval: int = 10)
```

A `Controller` (see [control.md](control.md#controller-abc)) whose
behaviour is authored at runtime. Skills: the idle `StandSkill` under
`PlanningController.IDLE` (`"__idle__"`) and, while one runs, the active
`CompositeSkill` under `PlanningController.PROGRAM` (`"program"`).

| Member | Behaviour |
|--------|-----------|
| `plan(state, last: PlanOutcome \| None) -> str \| None` | **override this.** Return the next program text, or `None` to idle. `last` is the outcome of the program that just finished, `None` on the first call and after an idle period |
| `decide(state)` | every `decision_interval` ticks: if the active program is finished, build its `PlanOutcome`, append to `history`, drop it and activate idle; then if nothing runs, call `plan()` and install what it returns |
| `program` | the active `CompositeSkill`, or `None` |
| `history` | every `PlanOutcome`, finished or failed to compile |
| `idle` | `program is None` (holding the stand pose) |
| `last_outcome` | `history[-1]` or `None` |

Installing compiles with `device=self.robot.device`, calls `setup(robot)`,
registers the program under `"program"`, resets it for all envs and makes
it active. `CompileError` and `GrammarError` do not crash the loop: the
outcome is recorded with `compile_error`, a line is printed, the controller
stays idle and `plan()` is asked again at the next decision tick, with
`last=None` (read `self.last_outcome.compile_error` to see what went wrong).
This is the repair feedback path an LLM planner needs.

Between the tick a program finishes and the next decision tick the finished
program keeps returning the hold pose.

## Built-in cards and `make_go2_library`

| Card | Interface | Accepts | Parameters (default, range) | `fail_when` | Constraints |
|------|-----------|---------|-----------------------------|-------------|-------------|
| `STAND_CARD` `stand` | motor | — | — | `fallen(0.15)` | — |
| `WALK_CARD` `walk` | motor | `velocity` | `vx` (0, [-1, 2]) m/s, `vy` (0, [-0.5, 0.5]) m/s, `vyaw` (0, [-1.5, 1.5]) rad/s | `tipped(0.9)`, `fallen(0.18)` | same as the ranges |
| `AVOID_CARD` `avoid` | command:velocity | — | — | `blocked(0.22)` | — |
| `FORWARD_CARD` `forward` | command:velocity | — | `distance` (1, [0, 20]) m, `speed` (0.5, [0.05, 1]) m/s | — | — |
| `BACKWARD_CARD` `backward` | command:velocity | — | `distance` (1, [0, 20]) m, `speed` (0.5, [0.05, 1]) m/s | — | — |
| `GOTO_CARD` `goto` | command:velocity | — | `x` (0, [-50, 50]) m, `y` (0, [-50, 50]) m, `speed` (0.5, [0.05, 1]) m/s | — | — |
| `SLAM_CARD` `slam` | command:velocity | — | — | — | — |

None of them has a machine-checked `success_when`: `stand` and `walk` run
until a modifier ends them; `forward`, `backward` and `goto` complete
through `TrajectoryTrackingSkill.success_flags`; `avoid` and `slam` never
complete on their own.

```python
make_go2_library(walk_policy, avoid_policy=None, lidar=None, cpg=None,
                 avoid_deltas=(0.8, 0.5, 1.5), obs_max_range: float = 4.0) -> SkillLibrary
```

| Registered | Skill factory | Requires |
|------------|---------------|----------|
| `stand` | `StandSkill` | always |
| `walk` | `CPGLocomotionSkill(walk_policy, cpg=cpg)` | always |
| `forward`, `backward`, `goto` | `TrajectoryTrackingSkill(mode)` | always (pose feedback, no policy) |
| conditions `blocked(d)`, `clear(d)` | closures over `lidar.read()` | `lidar` |
| `slam` | `SlamSkill(lidar)` with default `SlamConfig` | `lidar` |
| `avoid` | `LidarAvoidanceSkill(avoid_policy, lidar, deltas=avoid_deltas, obs_max_range=obs_max_range)` | `lidar` and `avoid_policy` |

`walk_policy` and `avoid_policy` are the callables described in
[control.md](control.md#cpglocomotionskill) (see
`domo.checkpoints.load_locomotion_policy`); `cpg`, `avoid_deltas` and
`obs_max_range` must match the values the policies were trained with.

```python
lib = make_go2_library(lambda o: torch.zeros(o.shape[0], 12))   # no lidar
lib.names()                      # ['backward', 'forward', 'goto', 'stand', 'walk']
```

## How to

### Write a PlanningController

Programs are authored **inside `plan()`**, from the state and the previous
outcome; they are never CLI flags or configuration
([conventions.md](../conventions.md)). Host the controller in a loop
exactly like any other controller.

```python
import random
from domo.skills import PlanningController, make_go2_library

class ExploreMission(PlanningController):
    """Wander legs with avoidance; rest after a failure; stop after N legs."""

    def __init__(self, library, legs: int = 6):
        super().__init__(library, decision_interval=10)
        self.legs, self.done_legs, self.fails = legs, 0, 0

    def plan(self, state, last):
        if last is not None:
            if last.compile_error:
                raise RuntimeError(last.compile_error)     # a researcher bug
            if last.succeeded:
                self.fails, self.done_legs = 0, self.done_legs + 1
            else:
                self.fails += 1
                if "FAILURE (safety" in "\n".join(last.trace) and self.fails < 3:
                    return "stand.for(2)"                  # cool off, then retry
                return None                                # give up → idle
        if self.done_legs >= self.legs:
            return None
        if self.done_legs % 2 == 0:
            return "(avoid @ walk(vx=0.5)).for(6) | stand.for(1)"
        return f"walk(vyaw={random.choice([-0.8, 0.8])}).for(2)"

library = make_go2_library(walk_policy, avoid_policy, world.lidar)
mission = ExploreMission(library)
mission.setup(world.robot)
loop = world.make_loop(mission)
loop.reset()
loop.run(5000)
for outcome in mission.history:
    print(outcome.program, outcome.succeeded)
```

**Persistent SLAM.** The `slam` card maps per leg because
`LayerNode.enter()` resets it. For a mission-wide map, own the `SlamSkill`
in the controller and tick it from `update()`; the programs then need no
`slam @`:

```python
from domo.control import SlamConfig, SlamSkill

class HouseWanderer(PlanningController):
    def __init__(self, library, slam: SlamSkill):
        super().__init__(library, decision_interval=5)
        self.slam = slam                      # set up and reset by the caller

    def update(self, state, dt):
        self.slam.update_command(state, dt)   # maps every tick, every leg
        return super().update(state, dt)

    def plan(self, state, last):
        if self.slam.coverage()[0] > 0.6:
            return None
        return "(avoid @ walk(vx=0.4)).for(6) | walk(vyaw=0.8).for(2)"

slam = SlamSkill(world.lidar, SlamConfig(resolution=0.1, half_extent=8.0))
slam.setup(world.robot)
slam.reset_idx(all_envs(world.robot))
controller = HouseWanderer(library, slam)
```

The map is not among the controller's `skills`, so `Controller.reset_idx`
does not wipe it; reset it yourself when the mission restarts.

### Add a card to the library

Four steps: write the skill (a `Skill` or `CommandSkill`,
[control.md](control.md#how-to)), describe it in a `SkillCard`, register a
factory, and register any sensor-bound condition the card or programs
need. Nothing else in the stack changes.

```python
import torch
from domo.skills import CMD_VELOCITY, MOTOR, ParamSpec, SkillCard, make_go2_library
from domo.control import LearnedJointSkill

# 1. A learned joint-space skill (an M3 output) as a motor card.
GETUP_CARD = SkillCard(
    name="getup",
    description="Stand back up from any fallen posture (learned policy).",
    interface=MOTOR,
    preconditions=["robot on the ground or tipped"],
    effects=["robot upright in nominal stance"],
    success_when=["upright"],            # must be valid condition syntax
    max_duration_s=6.0,                  # failsafe → FAILURE
)

# 2. A command skill with grammar parameters (FaceYawSkill from control.md).
FACE_CARD = SkillCard(
    name="face",
    description="Turn in place to an absolute heading; layer on walk: 'face(yaw=1.57) @ walk'.",
    interface=CMD_VELOCITY,
    params=[ParamSpec("yaw", "target heading, world frame", default=0.0,
                      range=(-3.15, 3.15), unit="rad")],
    preconditions=["layered on a velocity motor skill"],
    success_when=["heading within tolerance"],   # descriptive only
)

library = make_go2_library(walk_policy, avoid_policy, lidar)

# 3. Factories return a FRESH instance each time.
library.register(GETUP_CARD, lambda: LearnedJointSkill(getup_policy, getup_obs,
                                                       action_scale=0.25, name="getup"))
library.register(FACE_CARD, lambda: FaceYawSkill())

# 4. Conditions the cards use. State-only ones close over nothing;
#    sensor-bound ones close over the sensor object.
def upright(limit: float = 0.3):
    def cond(state, t_s):
        return state.base_euler[:, :2].abs().amax(dim=1) < limit
    return cond

def bright(threshold: float = 0.5):
    def cond(state, t_s):
        return light_sensor.read() > threshold        # [N]
    return cond

library.register_condition("upright", upright)
library.register_condition("bright", bright)

program = library.compile("getup >> (face(yaw=1.57) @ walk).until(bright(0.8))",
                          device=robot.device)
```

The card is what the planner sees (`library.describe()`), so its
description, preconditions and effects should say *when* to use the skill,
not how it works.

## Gotchas

* **Unknown conditions raise `KeyError`, not `CompileError`.**
  `library.compile("walk.until(flying)")` raises the registry's `KeyError`,
  and `PlanningController._install` catches only `CompileError` and
  `GrammarError`, so a program naming an unregistered condition crashes the
  loop. Validate condition names against `library.conditions.names()`
  before returning them from `plan()`.
* **After a compile error, `plan()` gets `last=None`.** The error is in
  `self.last_outcome.compile_error` (and in `history`); `last` only carries
  outcomes of programs that ran.
* **`NUM` is `-?\d+\.?\d*`.** `.5` and `1e3` are grammar errors; write
  `0.5` and `1000`. `.repeat(n)` accepts any `NUM` and truncates with
  `int(float(n))`, so `.repeat(2.7)` loops twice.
* **Modifiers bind tighter than `@`.** `avoid @ walk.for(4)` attaches
  `.for` to `walk` and is rejected at compile time; write
  `(avoid @ walk).for(4)`. The compiler's hint quotes the layer text as
  parsed, so `'(avoid @ walk(vx=0.5).for(3)).for(...)'` is not itself a
  valid fix; move the modifier outside.
* **`SUCCESS` is registered one tick late.** A container acts on a child's
  success at the start of its next update; `FAILURE` propagates the same
  tick through sequences and modifiers. Tests that count ticks need one
  extra update per transition.
* **All envs, any env.** `SUCCESS` requires every env, `FAILURE` any env,
  and the program counter is global. Compositions are for `n_envs = 1`.
* **`CompositeSkill.reset_idx` restarts the whole program** even when
  `envs_idx` is a subset, and clears the trace.
* **`LayerNode.enter()` resets the top skill and the motor skill for all
  envs.** `slam @ leg @ walk` starts a fresh map on every leg (host
  `SlamSkill` persistently instead), and `walk`'s oscillators restart on
  every leg that uses it.
* **Motor skills are shared, command skills are not.** One `walk`
  instance per program regardless of how many times it appears; every `@`
  makes a new `goto`/`avoid`/`slam` instance.
* **Override layers ignore the leaf's parameters.**
  `goto(...) @ walk(vx=0.9)` compiles, but `vx=0.9` is cancelled by the
  override; only the card clamp still applies.
* **`success_when` on a command card is not compiled.** It is prompt text
  only; completion comes from `success_flags`. On a motor card it must be
  valid condition syntax or `parse_condition` raises `GrammarError` at
  compile time.
* **`moved(d)` re-latches when `t_s` goes backwards.** It has no reset
  hook; the heuristic detects node re-entry through the node-local clock.
* **`TrajectoryTrackingSkill.configure` ignores unknown keys.** The card's
  `params` are the only validation of parameter names.
* **`describe()` lists conditions from the registry**, so a condition
  registered after the library was built appears in the prompt only if you
  call `describe()` again.
