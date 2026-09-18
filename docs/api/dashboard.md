# `domo.dashboard` — live telemetry for the digital twin

`domo/dashboard.py` is a single-file web dashboard for watching a twin or
evaluation run in a browser: a 3D view of the robot in its scene, the SLAM
occupancy map, the lidar sweep, the point cloud, velocity and height
charts, and three buttons (pause, reset, stop) that reach back into the
simulation loop. It is standard library only (`http.server`, `urllib`,
`threading`, `json`); the page itself is one HTML constant embedded in the
module.

**It is never on the training path.** The dashboard exists for the twin
runtime and for evaluation ([architecture.md](../architecture.md)); no
task, trainer or `VecTask` imports it, and the examples that use it
(`examples/slam/slam_demo.py`, `examples/slam/slam_replica_house.py`)
import `domo.dashboard` lazily, only when `--dashboard PORT` or
`--dashboard-url URL` is given, so no web code loads otherwise. Keep it
that way when wiring it into a new example.

Failure modes (blank page, no 3D, `NameError` on `python -m
domo.dashboard`) are in
[troubleshooting.md, Dashboard](../troubleshooting.md#dashboard).

## Architecture

```
  browser ◀── GET /, /state, /scene, /frame.jpg, /assets/… ──┐
          ──▶ GET /cmd?c=pause|reset|stop ─────────────────────┤
                                                              ▼
                                                    DashboardServer
                                                   (ThreadingHTTPServer,
                                                    one _Handler per request)
                                                              │
                                                        TelemetryHub
                                                   latest state · frame ·
                                                   scene · pending command
                                                              ▲
     sim process ── DashboardClient ── daemon thread ─────────┤
       publish()      ref swap          POST /ingest, /scene, /frame
       poll_command() ref swap          GET  /cmd-poll
```

Five classes, in the order the module defines them:

| Class | Role |
|-------|------|
| `TelemetryHub` | thread-safe latest-value store: the state dict, the last JPEG and its id, the scene manifest and asset roots, one pending command. Only the latest snapshot is kept (a live view, not a log), so every setter is an O(1) reference swap under a short lock. |
| `_Handler` | `BaseHTTPRequestHandler` with one method per route; `log_message` is silenced because the page polls at about 7 Hz. Every response carries `Cache-Control: no-store` and `Access-Control-Allow-Origin: *`. |
| `DashboardServer` | `ThreadingHTTPServer` bound to `127.0.0.1` with `daemon_threads = True` and `allow_reuse_address = True`, carrying the hub and the rendered page. Built by `make_server(port, hub, title)`. |
| `DashboardClient` | what the simulation uses when the server runs in another process. `publish` and `poll_command` never touch the network on the sim thread. |
| `Dashboard` | the server hosted inside the sim process, same `start / set_scene / publish / poll_command / stop` interface, `publish` writes straight into the hub. |

### Threading model, and why `publish` is O(1)

The sim thread and the network only meet through lock-guarded slots. In
`DashboardClient`, `publish(state, frame)` stores the references and
returns; `set_scene` stores the manifest; `poll_command` reads and clears
the pending command. A daemon thread (`_run`) loops at `post_hz`: it takes
everything pending in one swap, POSTs the scene if one is queued, the
state if one is queued, the frame (JPEG-encoded on that thread) if one is
queued, then GETs `/cmd-poll` and sleeps for the period. Between two
posts only the latest snapshot survives; stale ones are dropped on
purpose.

Every request has a 1 s timeout and every transport error is swallowed
and logged at DEBUG (`POST /ingest dropped: ...`), so a server that is not
running, hangs or goes away mid-run costs the sim nothing: publishes are
dropped, `poll_command` returns `None`, the thread stays alive
(`test_client_survives_missing_server`).

The in-process `Dashboard` has the same interface but one cost: with a
camera frame, `_encode_jpeg` (numpy plus Pillow, imported lazily; Pillow is
not a declared dependency) runs on the sim thread. That is why the camera
panel is opt-in (`--dashboard-camera`) and why the decoupled client is
the recommended path. Without a frame `Dashboard.publish` is a hub write.

The server side is also O(1) per request: `TelemetryHub.publish` swaps
the dict, `get_state` copies it and adds bookkeeping keys, `pop_command`
is read-and-clear so a button press reaches the sim exactly once.

## Starting it

### Decoupled (recommended): two processes

```
# shell 1: the server, bound to 127.0.0.1
python -m domo.dashboard --port 8080 [--title "DOMO twin"]

# shell 2: the simulation
python examples/slam/slam_demo.py --headless --device cpu --dashboard-url http://127.0.0.1:8080
```

In code:

```python
from domo.dashboard import DashboardClient

dash = DashboardClient("http://127.0.0.1:8080", post_hz=20.0).start()
dash.set_scene(manifest, roots)              # once, before or after start()
for i in range(steps):
    cmd = dash.poll_command()                # "pause" | "reset" | "stop" | None
    ...
    state = loop.step()
    if i % 10 == 0:
        dash.publish(snapshot_dict, frame=None)   # non-blocking
dash.stop()                                  # sets the stop event; the daemon thread exits
```

`start()` returns `self` and prints the server command to run if none is
listening. Open `http://127.0.0.1:8080` in a browser; until a sim pushes,
the page shows step `0` and `_has_scene: false`.

### In-process

```
python examples/slam/slam_demo.py --headless --device cpu --dashboard 8080 [--dashboard-camera]
```

```python
from domo.dashboard import Dashboard

dash = Dashboard(port=8080, title="DOMO SLAM — arena").start()
dash.set_scene(manifest, roots)
dash.publish(snapshot_dict, frame=camera.render())   # frame optional; encoded here
dash.poll_command()
dash.stop()                                           # server.shutdown() + close
```

`serve(port=8080, title="DOMO twin")` is the blocking entry point behind
`python -m domo.dashboard`; `make_server(port, hub, title)` builds an
unstarted `DashboardServer` (port `0` for an ephemeral one, as the tests
do).

### Signatures

```python
class TelemetryHub:
    def publish(self, state: dict, frame: bytes | None = None) -> None
    def set_frame(self, jpeg: bytes) -> None
    def get_state(self) -> dict                  # + _uptime, _frame_id, _has_scene
    def get_frame(self) -> tuple[bytes | None, int]
    def set_command(self, cmd: str) -> None
    def pop_command(self) -> str | None          # read-and-clear
    def set_scene(self, scene: dict | None, roots: dict[str, str] | None = None) -> None
    def get_scene(self) -> dict | None
    def resolve_asset(self, name: str, rel: str) -> str | None

def make_server(port: int, hub: TelemetryHub, title: str) -> DashboardServer
def serve(port: int = 8080, title: str = "DOMO twin") -> None

class DashboardClient:
    def __init__(self, base_url: str, post_hz: float = 20.0)
    def start(self) -> DashboardClient
    def set_scene(self, scene: dict, roots: dict | None = None) -> None
    def publish(self, state: dict, frame=None) -> None
    def poll_command(self) -> str | None
    def stop(self) -> None

class Dashboard:
    def __init__(self, port: int = 8080, title: str = "DOMO twin")
    # start / set_scene / publish / poll_command / stop as above
```

Module constants: `BIND_HOST = "127.0.0.1"`, `COMMANDS = ("pause",
"reset", "stop")`, `REQUEST_TIMEOUT_S = 1.0`, `JPEG_QUALITY = 80`.

## Routes

All bound to loopback. `_send` adds `Content-Type`, `Content-Length`,
`Cache-Control: no-store` and `Access-Control-Allow-Origin: *` to every
response. Unknown paths, GET or POST, are `404 not found`.

| Method | Path | Request | Response |
|--------|------|---------|----------|
| GET | `/`, `/index.html` | — | `200`, the page with `__TITLE__` substituted |
| GET | `/state` | — | `200` JSON: the latest telemetry dict plus `_uptime` (s since the hub was created, one decimal), `_frame_id` (bumps per camera frame so the page can skip repeats), `_has_scene` (bool) |
| GET | `/scene` | — | `200` JSON manifest, or `204` when none was set |
| GET | `/frame.jpg` | — | `200 image/jpeg`, or `204` when no frame was pushed |
| GET | `/cmd?c=CMD` | `CMD` in `pause`, `reset`, `stop` | `200 ok` and the command is stored; anything else `400 bad command` |
| GET | `/cmd-poll` | — | `200` `{"cmd": "pause" \| "reset" \| "stop" \| null}`, read-and-clear |
| GET | `/assets/<name>/<rel>` | — | `200` the file under the root registered as `<name>`, content type from the extension (`.urdf`/`.xml` → `application/xml`, `.dae` → `model/vnd.collada+xml`, `.stl`, `.glb`, `.gltf`, `.obj`, `.mtl`, `.png`, `.jpg`, else `application/octet-stream`); `404` for an unknown root, a missing file, a directory, or any path that normalises outside the root |
| POST | `/ingest` | JSON object | `200 ok`; replaces the snapshot. A body that is not JSON, or not a dict, is dropped (DEBUG log) and the previous snapshot stays; still `200` |
| POST | `/scene` | JSON `{"scene": manifest, "roots": {name: abs_dir}}` | `200 ok`; a body that is not a JSON object is dropped |
| POST | `/frame` | raw JPEG bytes | `200 ok`; becomes the latest frame, `_frame_id` increments |

The asset route serves whole mesh directories, so the traversal guard
matters: `resolve_asset` normalises `root/rel` and refuses anything that
is not strictly inside the registered root
(`test_resolve_asset_guards_traversal`,
`test_assets_route_serves_registered_root_only`).

## Telemetry payload

The page reads whatever `/state` returns; every key is optional except
`t`. The hub starts with `{"t": 0}`. `examples/slam/slam_viz.py::slam_snapshot(step, state, slam, lidar=None, cloud_pts=500, status="running")`
builds a complete one from a `RobotState` and a `SlamSkill` (values
rounded, cloud subsampled to `cloud_pts` points, occupancy grid max-pooled
to about 60 rows) and is cheap enough to call at a few Hz.

| Key | Type | Used by |
|-----|------|---------|
| `t` | int, control step | header badge; the publish-rate badge is derived from consecutive `t` values |
| `status` | `"running"` \| `"paused"` \| `"stopped"` | status badge colour and the pause button label (`⏸ pause` / `▶ resume`); missing means running |
| `pose` | `[x, y, yaw]` | header badge (x, y shown) |
| `base` | `[x, y, z, qw, qx, qy, qz]`, quaternion **wxyz** as everywhere in DOMO | 3D viewer: robot base pose (the page reorders to three.js xyzw) |
| `dof` | `[q_0 … q_n]` in the scene manifest's `dof_names` order | 3D viewer: joint angles via `setJointValue` |
| `vel` | `[vx, vy, vyaw]` | velocity chart, fixed range ±1.5 |
| `height` | float, m | height/coverage chart |
| `coverage` | float, percent | header badge and chart |
| `lidar` | `[range per azimuth sector]`, sector `i` at angle `i / n · 2π` | top-down lidar panel, normalised to the max range in the scan |
| `map` | list of equal-length row strings of `#` occupied, `.` free, space unknown | occupancy panel |
| `cloud` | `[[x, y, z], …]` | top-down point-cloud panel, coloured by z |
| `bounds` | `[x_min, x_max, y_min, y_max]` | extent of the cloud panel; required for the cloud to draw |

Chart history is kept in the browser (last 600 samples per series), so a
page reload starts the charts empty.

## Scene manifest

Pushed once with `set_scene(scene, roots)`; the page fetches `/scene` as
soon as `/state` reports `_has_scene` and loads it exactly once per page
load (reload the page after changing the scene).

```python
scene = {
    "boxes": [{"size": [sx, sy, sz], "pos": [x, y, z],
               "quat": [w, x, y, z],          # optional, wxyz
               "color": "#8b949e"}, ...],     # optional, any CSS colour
    "robot": {"urdf": "/assets/robot/urdf/go2.urdf",   # served by /assets
              "dof_names": [...]},                     # order of `dof` in telemetry
    "up": "z",
}
roots = {"robot": "/abs/path/to/go2"}       # the dir that contains urdf/ and meshes/
```

`roots` maps the `<name>` segment of `/assets/<name>/<rel>` to an
absolute directory. The URDF's mesh references are relative, so the root
must be the directory the URDF resolves them against, i.e. the one
holding both `urdf/` and `meshes/`. `slam_viz.go2_robot_manifest(spec)`
returns `(robot_dict, root_dir)` for a `RobotSpec`, locating Genesis'
bundled asset directory when `spec.urdf_path` is relative. The example
adds a ground plane and a grid itself; `boxes` is for obstacles and walls
(`slam_demo.arena_manifest_boxes` mirrors `build_arena` box for box).

## The page

Dark, monospace, a sticky header and a four-column card grid (two columns
below 960 px). Everything is inline; there is no build step.

**Header badges:** `step`, `pub … Hz` (publish rate, from `t` deltas
between polls), `pose`, `coverage`, the connection badge (`● live` green
when `/state` answers, `○ no server` red when it does not), and `status`
(green running, amber paused, red stopped).

**Buttons:** `⏸ pause` / `▶ resume`, `⟳ reset`, `⏹ stop` (with a
confirm dialog). Each is a `fetch('/cmd?c=…')`; the page does nothing
else. The semantics are owned by the example loop that polls the
command, and `slam_demo.run` is the reference:

| Command | What the example loop does |
|---------|----------------------------|
| `pause` | toggles a `paused` flag; while paused the sim does not step but keeps publishing with `status="paused"` every 50 ms so the page stays live; a second `pause` resumes |
| `reset` | `loop.reset()`, `slam.reset_idx(...)`, step counter back to 0 |
| `stop` | breaks out of the run; the example then publishes a final snapshot with `status="stopped"`, prints its report and keeps serving in `hold_dashboard` until Ctrl+C or another `stop` |

A command is delivered once (`/cmd-poll` clears it) and only when the
loop asks; a press while the loop is inside a long `scene.step()` is
picked up at the next poll.

**Panel chips:** `simulation`, `slam map`, `lidar`, `point cloud`,
`velocity`, `height/coverage`, `raw telemetry`. All are on by default
except raw telemetry (the full `/state` JSON, pretty-printed). A chip or
a card's `✕` toggles the card; the set of visible panels is persisted in
`localStorage['domo_panels']` and restored on the next load. Only visible
panels are redrawn on each poll, and the 3D renderer only renders while
the simulation card is visible.

**Polling:** `/state` every 150 ms from a plain `<script>`; the page
never uses websockets or server push.

## The 3D viewer

The simulation card is a `<script type="module">` that dynamically
imports three.js 0.160.0 (`OrbitControls`, `ColladaLoader`) and
`urdf-loader` 0.12.1 through an import map pointing at `unpkg.com`. The
browser therefore needs internet access even though the server is local.
The import is wrapped in `try/catch`: a CDN failure writes a message into
the 3D panel and returns, and because the telemetry code is a separate
plain `<script>` that runs first, the rest of the page is unaffected
(`test_page_splits_core_and_3d_scripts`).

Z is up (`THREE.Object3D.DEFAULT_UP = (0, 0, 1)`, the grid rotated into
the XY plane); drag to orbit, scroll to zoom. The robot is loaded with
`URDFLoader` and `ColladaLoader` from `/assets/...`; the Go2 COLLADA
meshes are tens of megabytes, so the first load takes seconds and the
panel shows `loading robot…` meanwhile. Box quaternions and the `base`
quaternion are converted from wxyz to three.js xyzw in the page.

The in-browser rendering has not been verified in a sandboxed
environment; the tests cover the routes, the payloads and the script
layout, not WebGL.

## Gotchas

* **`_DASHBOARD_HTML` must stay before the `__main__` guard, and the
  guard must be the last statement.** `python -m domo.dashboard` executes
  the module top to bottom; `make_server` reads the constant. Enforced by
  `test_html_constant_precedes_main_guard`, which also checks that the
  page contains no `"""` (it lives in an `r"""` literal).
* **Ctrl+C is a hard kill in dashboard examples.** `scene.build()` and
  `scene.step()` hold the GIL and defer `KeyboardInterrupt`, so the
  examples set `signal.signal(signal.SIGINT, signal.SIG_DFL)` at the top
  of `main()`. Nothing is flushed on Ctrl+C; the stop button is the
  graceful path (report, PNG, final `stopped` snapshot).
* **The camera must be added before `scene.build()`.** `--dashboard-camera`
  calls `scene.add_camera` in `build_world`; it is a build-time operation.
* **No history.** A reconnecting page sees the latest snapshot only;
  charts and the SLAM map fill in from that point. Log to disk in the
  example if you need a record.
* **Loopback only, no authentication.** `BIND_HOST` is `127.0.0.1` and
  CORS is `*`, which is fine for a local viewer and would not be for
  anything else.
* **Publish cadence is the caller's choice.** `slam_demo` publishes every
  10 control steps (about 5 Hz at 50 Hz control) and on every iteration
  while paused; `post_hz` only bounds the client's background thread.
* **Any scene change needs a page reload**; the manifest is fetched once
  per page load.

## Tests

`tests/test_dashboard_server.py`, 17 tests, engine-free, no network
beyond loopback (a real `DashboardServer` on an ephemeral port):

| Group | What is covered |
|-------|-----------------|
| `TelemetryHub` (5) | bookkeeping keys, snapshot replacement and frame ids, read-and-clear commands, `_has_scene`, asset traversal guard |
| HTTP routes (7) | titled page and headers, `204` for missing scene/frame, `/cmd` validation and relay, `404`s, POST routes feeding the hub, malformed POSTs dropped, asset route confined to its root |
| `DashboardClient` (2) | full round trip including latest-only delivery and one-shot commands; survival with no server listening |
| Module invariants (3) | HTML constant before the `__main__` guard, core/3D script split, `_encode_jpeg` from a float frame (skipped without numpy or Pillow) |

```
pytest tests/test_dashboard_server.py -q
```
