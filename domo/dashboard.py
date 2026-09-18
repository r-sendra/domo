"""
Decoupled live dashboard for the digital twin / EVALUATION (never training).

Two processes, launched separately — the sim never hosts the UI:

  1. the DASHBOARD SERVER (this module, standalone):
         python -m domo.dashboard --port 8080
     Serves the web UI, holds the latest telemetry + scene manifest, relays
     control commands, and serves static robot/scene assets for the 3D viewer.

  2. the SIMULATION, which pushes to it via `DashboardClient`:
         dash = DashboardClient("http://127.0.0.1:8080").start()
         dash.set_scene(manifest, roots)          # once
         for step in loop:
             sim.step()
             dash.publish(state_dict)             # non-blocking
             if dash.poll_command() == "stop": ...

Why it still never slows the sim: `publish()` only swaps the latest snapshot
into a slot (microseconds); a BACKGROUND thread does all network I/O — POSTing
the latest state (stale frames dropped) and polling for commands. If the server
is down, posts fail silently; the sim is unaffected.

`Dashboard` (server hosted inside the sim process) is kept for convenience and
shares the same publish/poll_command/set_scene interface.

HTTP API (all bound to 127.0.0.1; every response carries `Cache-Control:
no-store` and `Access-Control-Allow-Origin: *`):

    GET  /, /index.html   the self-contained dashboard page (`_DASHBOARD_HTML`)
    GET  /state           latest telemetry dict as JSON, plus `_uptime` (s),
                          `_frame_id` (bumps per camera frame) and `_has_scene`
    GET  /scene           3D scene manifest as JSON, or 204 when none was set
    GET  /frame.jpg       latest camera JPEG, or 204 when none was pushed
    GET  /cmd?c=CMD       browser → sim control; CMD in {pause, reset, stop}
    GET  /cmd-poll        sim ← pending command: {"cmd": CMD | null}, read-and-clear
    GET  /assets/<name>/<rel>   static files under the asset root registered
                          as <name> (URDF + meshes for the 3D viewer)
    POST /ingest          telemetry dict (JSON) → replaces the latest snapshot
    POST /scene           {"scene": manifest, "roots": {name: abs_dir}}
    POST /frame           raw JPEG bytes → latest camera frame

Telemetry payload (what the page reads; all keys optional except `t`):
    t         int    control step
    status    str    "running" | "paused" | "stopped"  (badge + button label)
    pose      [x, y, yaw]                 header badge
    base      [x, y, z, qw, qx, qy, qz]   3D viewer: base pose (wxyz quaternion)
    dof       [q_0 … q_n]                 3D viewer: joint angles in `dof_names` order
    vel       [vx, vy, vyaw]              velocity chart
    height    float                       height chart
    coverage  float (percent)             header badge + chart
    lidar     [range per azimuth sector]  top-down lidar panel
    map       [row strings of '#', '.', ' ']   occupancy panel
    cloud     [[x, y, z], …]              top-down point-cloud panel
    bounds    [x_min, x_max, y_min, y_max]     cloud panel extent

Scene manifest (POST /scene → GET /scene):
    boxes   [{size: [sx, sy, sz], pos: [x, y, z], quat?: [w, x, y, z], color?: css}]
    robot   {urdf: "/assets/<name>/<rel>.urdf", dof_names: [...]}
    up      "z"
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

__all__ = [
    "Dashboard",
    "DashboardClient",
    "DashboardServer",
    "TelemetryHub",
    "make_server",
    "serve",
]

_log = logging.getLogger(__name__)

# Loopback only: the dashboard is a local viewer, never exposed on the network.
BIND_HOST = "127.0.0.1"
# Control commands the page may send; anything else is a 400.
COMMANDS = ("pause", "reset", "stop")
# Network calls from the client's background thread are short so a hung server
# cannot stall command polling for long.
REQUEST_TIMEOUT_S = 1.0
JPEG_QUALITY = 80

# Content types for the static asset route (three.js loaders sniff these).
_CONTENT_TYPES = {
    ".urdf": "application/xml", ".xml": "application/xml",
    ".dae": "model/vnd.collada+xml", ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json", ".obj": "text/plain", ".mtl": "text/plain",
    ".stl": "model/stl", ".png": "image/png", ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

class TelemetryHub:
    """
    Thread-safe latest-value store shared by the server's handler threads.

    Only the LATEST snapshot is kept — the dashboard is a live view, not a
    log — so every setter is an O(1) reference swap under a short lock.
    Handler threads (one per request under `ThreadingHTTPServer`) and the
    publishing thread all go through this lock; nothing else is shared.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._state: dict = {"t": 0}
        self._frame: bytes | None = None   # latest camera frame, JPEG bytes
        self._frame_id = 0                 # bumps per frame so the page can skip repeats
        self._cmd: str | None = None       # pending control command
        self._scene: dict | None = None    # 3D scene manifest
        self._roots: dict[str, str] = {}   # asset name → absolute dir (for /assets)
        self._t0 = time.time()

    def publish(self, state: dict, frame: bytes | None = None) -> None:
        with self._lock:
            self._state = state
            if frame is not None:
                self._frame = frame
                self._frame_id += 1

    def set_frame(self, jpeg: bytes) -> None:
        with self._lock:
            self._frame = jpeg
            self._frame_id += 1

    def get_state(self) -> dict:
        """Latest snapshot plus server-side bookkeeping keys (`_uptime`, ...)."""
        with self._lock:
            s = dict(self._state)
            s["_uptime"] = round(time.time() - self._t0, 1)
            s["_frame_id"] = self._frame_id
            s["_has_scene"] = self._scene is not None
            return s

    def get_frame(self) -> tuple[bytes | None, int]:
        with self._lock:
            return self._frame, self._frame_id

    def set_command(self, cmd: str) -> None:
        with self._lock:
            self._cmd = cmd

    def pop_command(self) -> str | None:
        """Read-and-clear, so a command is delivered to the sim exactly once."""
        with self._lock:
            c, self._cmd = self._cmd, None
            return c

    def set_scene(self, scene: dict | None, roots: dict[str, str] | None = None) -> None:
        with self._lock:
            self._scene = scene
            if roots:
                self._roots = dict(roots)

    def get_scene(self) -> dict | None:
        with self._lock:
            return self._scene

    def resolve_asset(self, name: str, rel: str) -> str | None:
        """Absolute path for /assets/<name>/<rel>, or None (with traversal guard)."""
        with self._lock:
            root = self._roots.get(name)
        if not root:
            return None
        base = os.path.normpath(root)
        p = os.path.normpath(os.path.join(base, rel))
        # Anything that normalises outside the registered root is refused —
        # the route serves whole mesh directories, so `..` must not escape.
        if p != base and not p.startswith(base + os.sep):
            return None
        return p if os.path.isfile(p) else None


# ---------------------------------------------------------------------------
# HTTP handler + server
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    """Routes documented in the module docstring. One instance per request."""

    def log_message(self, *args):
        pass                    # the page polls at ~7 Hz; stdlib access logs would drown stdout

    @property
    def hub(self) -> TelemetryHub:
        return self.server.hub          # type: ignore[attr-defined]

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            if body:
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass                # browser navigated away mid-response; nothing to do

    def _send_json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _send_text(self, code: int, text: bytes) -> None:
        self._send(code, text, "text/plain")

    # --- GET ---------------------------------------------------------------

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        path, query = u.path, urllib.parse.parse_qs(u.query)
        if path in ("/", "/index.html"):
            self._send(200, self.server.html, "text/html; charset=utf-8")  # type: ignore[attr-defined]
        elif path == "/state":
            self._send_json(self.hub.get_state())
        elif path == "/scene":
            self._get_scene()
        elif path == "/frame.jpg":
            self._get_frame()
        elif path == "/cmd":
            self._get_cmd(query)
        elif path == "/cmd-poll":
            self._send_json({"cmd": self.hub.pop_command()})
        elif path.startswith("/assets/"):
            self._get_asset(path[len("/assets/"):])
        else:
            self._send_text(404, b"not found")

    def _get_scene(self) -> None:
        scene = self.hub.get_scene()
        if scene:
            self._send_json(scene)
        else:
            self._send(204, b"", "application/json")

    def _get_frame(self) -> None:
        frame, _ = self.hub.get_frame()
        self._send(204 if frame is None else 200, frame or b"", "image/jpeg")

    def _get_cmd(self, query: dict) -> None:
        cmd = (query.get("c") or [""])[0]
        if cmd in COMMANDS:
            self.hub.set_command(cmd)
            self._send_text(200, b"ok")
        else:
            self._send_text(400, b"bad command")

    def _get_asset(self, rest: str) -> None:
        name, _, rel = rest.partition("/")
        fp = self.hub.resolve_asset(name, rel)
        if not fp:
            self._send_text(404, b"not found")
            return
        with open(fp, "rb") as f:
            body = f.read()
        ctype = _CONTENT_TYPES.get(os.path.splitext(fp)[1].lower(),
                                   "application/octet-stream")
        self._send(200, body, ctype)

    # --- POST --------------------------------------------------------------

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        n = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(n) if n else b""
        if path == "/ingest":
            self._post_ingest(body)
        elif path == "/scene":
            self._post_scene(body)
        elif path == "/frame":
            self.hub.set_frame(body)
            self._send_text(200, b"ok")
        else:
            self._send_text(404, b"not found")

    def _post_ingest(self, body: bytes) -> None:
        # A malformed snapshot is dropped, never surfaced: the sim's publisher
        # thread fires and forgets, so the only useful reaction is to keep the
        # previous snapshot and wait for the next one.
        try:
            state = json.loads(body)
        except ValueError:
            _log.debug("dropped /ingest body that is not JSON")
        else:
            if isinstance(state, dict):
                self.hub.publish(state)
            else:
                _log.debug("dropped /ingest payload that is not a dict")
        self._send_text(200, b"ok")

    def _post_scene(self, body: bytes) -> None:
        try:
            d = json.loads(body)
            self.hub.set_scene(d.get("scene"), d.get("roots"))
        except (ValueError, AttributeError):    # not JSON / not an object
            _log.debug("dropped malformed /scene body")
        self._send_text(200, b"ok")


def _encode_jpeg(frame) -> bytes:
    """uint8 [H, W, 3+] array → JPEG bytes (PIL imported lazily: optional path)."""
    import numpy as np
    from PIL import Image
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr[..., :3]).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


class DashboardServer(ThreadingHTTPServer):
    """One handler thread per request; daemon so the sim process can exit freely."""
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, hub: TelemetryHub, html: bytes):
        super().__init__(address, handler)
        self.hub = hub
        self.html = html


def make_server(port: int, hub: TelemetryHub, title: str) -> DashboardServer:
    html = _DASHBOARD_HTML.replace("__TITLE__", title).encode()
    return DashboardServer((BIND_HOST, port), _Handler, hub, html)


def serve(port: int = 8080, title: str = "DOMO twin") -> None:
    """Run the standalone dashboard server (blocking). `python -m domo.dashboard`."""
    hub = TelemetryHub()
    srv = make_server(port, hub, title)
    print(f"  [dashboard] serving http://{BIND_HOST}:{port}  "
          f"(open it; waiting for a sim to push telemetry) — Ctrl+C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


# ---------------------------------------------------------------------------
# Sim-side client: non-blocking push over the network (separate process).
# ---------------------------------------------------------------------------

class DashboardClient:
    """
    What the simulation uses to feed a standalone dashboard server. `publish()`
    and `poll_command()` never touch the network on the sim thread — a daemon
    thread POSTs the latest snapshot (dropping stale ones) and polls commands.

    Thread-safety: the sim thread and the background thread only meet through
    `_lock`-guarded slots (`_state`, `_frame`, `_scene`, `_cmd`); each is a
    plain reference swap, so the sim never waits on I/O.
    """

    def __init__(self, base_url: str, post_hz: float = 20.0):
        self.base = base_url.rstrip("/")
        self.period = 1.0 / max(post_hz, 1.0)
        self._lock = threading.Lock()
        self._state: dict | None = None
        self._frame = None
        self._cmd: str | None = None
        self._scene: dict | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> DashboardClient:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"  [dashboard] pushing telemetry to {self.base} (decoupled; "
              f"start it with:  python -m domo.dashboard --port "
              f"{self.base.rsplit(':', 1)[-1]})")
        return self

    def set_scene(self, scene: dict, roots: dict | None = None) -> None:
        with self._lock:
            self._scene = {"scene": scene, "roots": roots or {}}

    def publish(self, state: dict, frame=None) -> None:
        with self._lock:
            self._state = state
            if frame is not None:
                self._frame = frame

    def poll_command(self) -> str | None:
        with self._lock:
            c, self._cmd = self._cmd, None
            return c

    def stop(self) -> None:
        self._stop.set()

    # --- background thread ---------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:      # take everything pending in one swap
                st, self._state = self._state, None
                fr, self._frame = self._frame, None
                sc, self._scene = self._scene, None
            if sc is not None:
                self._post("/scene", json.dumps(sc).encode())
            if st is not None:
                self._post("/ingest", json.dumps(st).encode())
            if fr is not None:
                self._post("/frame", _encode_jpeg(fr), "image/jpeg")
            self._poll_command()
            time.sleep(self.period)

    def _poll_command(self) -> None:
        got = self._get("/cmd-poll")
        if not got:
            return
        try:
            cmd = json.loads(got).get("cmd")
        except (ValueError, AttributeError):    # garbage reply → no command
            cmd = None
        if cmd:
            with self._lock:
                self._cmd = cmd

    def _post(self, path: str, body: bytes, ctype: str = "application/json") -> None:
        try:
            req = urllib.request.Request(self.base + path, data=body,
                                         headers={"Content-Type": ctype},
                                         method="POST")
            urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S).read()
        except Exception as e:
            # Deliberately broad: this daemon thread must survive ANY transport
            # error (server down, timeout, half-closed socket) — dropping a
            # snapshot is the intended behaviour, crashing the sim is not.
            _log.debug("POST %s dropped: %s", path, e)

    def _get(self, path: str) -> bytes | None:
        try:
            return urllib.request.urlopen(self.base + path,
                                          timeout=REQUEST_TIMEOUT_S).read()
        except Exception as e:      # same rationale as _post
            _log.debug("GET %s failed: %s", path, e)
            return None


# ---------------------------------------------------------------------------
# In-process convenience (server hosted inside the sim); same interface.
# ---------------------------------------------------------------------------

class Dashboard:
    """Server in a daemon thread of the sim process; publish() goes straight to the hub."""

    def __init__(self, port: int = 8080, title: str = "DOMO twin"):
        self.port = port
        self.title = title
        self.hub = TelemetryHub()
        self._server: DashboardServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> Dashboard:
        self._server = make_server(self.port, self.hub, self.title)
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        daemon=True)
        self._thread.start()
        print(f"  [dashboard] live at http://{BIND_HOST}:{self.port}  "
              f"(open it in a browser; the sim runs independently)")
        return self

    def set_scene(self, scene: dict, roots: dict | None = None) -> None:
        self.hub.set_scene(scene, roots)

    def publish(self, state: dict, frame=None) -> None:
        # JPEG encoding runs here on the sim thread — the one cost of the
        # in-process variant, which is why the camera panel is opt-in.
        self.hub.publish(state, _encode_jpeg(frame) if frame is not None else None)

    def poll_command(self) -> str | None:
        return self.hub.pop_command()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


def main():
    ap = argparse.ArgumentParser(description="DOMO dashboard server (standalone)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--title", default="DOMO twin")
    a = ap.parse_args()
    serve(a.port, a.title)


# ---------------------------------------------------------------------------
# The dashboard page — 3D (three.js) sim viewer + sensor/SLAM panels, toggles,
# and pause/reset/stop controls. three.js loaded from a CDN (browser needs net).
#
# Placement matters: this constant is defined AFTER the functions that use it
# and BEFORE the `__main__` guard — `python -m domo.dashboard` executes the
# module top to bottom, so the guard must stay the LAST statement or
# `make_server` would hit a NameError.
# Two script blocks on purpose: telemetry/2D panels/controls live in a plain
# <script> that always runs; the three.js viewer is a separate module with a
# guarded dynamic import, so a CDN failure only blanks the 3D panel.
# ---------------------------------------------------------------------------
_DASHBOARD_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>__TITLE__</title>
<script type="importmap">
{"imports":{
  "three":"https://unpkg.com/three@0.160.0/build/three.module.js",
  "three/":"https://unpkg.com/three@0.160.0/",
  "three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/",
  "three/examples/jsm/":"https://unpkg.com/three@0.160.0/examples/jsm/",
  "urdf-loader":"https://unpkg.com/urdf-loader@0.12.1/src/URDFLoader.js"
}}
</script>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}
body{margin:0;background:#0d1117;color:#e6edf3;font:13px/1.4 ui-monospace,Menlo,monospace}
header{padding:10px 16px;background:#161b22;border-bottom:1px solid #30363d;position:sticky;top:0;z-index:5}
.top{display:flex;gap:16px;align-items:center;flex-wrap:wrap}
header h1{font-size:15px;margin:0;font-weight:600}
.badge{background:#21262d;border:1px solid #30363d;border-radius:6px;padding:3px 8px}
.ctl{display:flex;gap:6px;margin-left:auto}
.btn{background:#21262d;border:1px solid #30363d;color:#e6edf3;border-radius:6px;padding:4px 12px;cursor:pointer;font:inherit;font-size:12px}
.btn:hover{border-color:#8b949e}.btn.danger:hover{border-color:#f85149;color:#f85149}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:9px}.chips .lbl{color:#7d8590;align-self:center}
.chip{background:#21262d;border:1px solid #30363d;color:#7d8590;border-radius:14px;padding:4px 11px;cursor:pointer;font:inherit;font-size:12px;user-select:none}
.chip.on{background:#1f6feb22;border-color:#1f6feb;color:#cae1ff}
.wrap{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;padding:12px;align-items:start}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:10px;min-width:0}
.card.main{grid-column:span 2;grid-row:span 2}.card.hidden{display:none}
.card h2{font-size:12px;margin:0 0 8px;color:#7d8590;text-transform:uppercase;letter-spacing:.5px;display:flex;justify-content:space-between}
.card h2 .x{color:#586069;cursor:pointer}.card h2 .x:hover{color:#f85149}
canvas{width:100%;background:#0d1117;border-radius:6px;display:block}
#view3d{width:100%;height:60vh;min-height:340px;background:#0d1117;border-radius:6px;position:relative}
#view3d .msg{position:absolute;left:10px;top:10px;right:10px;color:#7d8590;font-size:12px}
pre{margin:0;white-space:pre-wrap;font-size:11px;color:#9da7b3;max-height:300px;overflow:auto}
@media(max-width:960px){.wrap{grid-template-columns:repeat(2,1fr)}.card.main{grid-column:1/-1;grid-row:auto}}
</style></head><body>
<header>
  <div class="top">
    <h1>__TITLE__</h1>
    <span class="badge">step <b id="t">–</b></span>
    <span class="badge">pub <b id="fps">–</b> Hz</span>
    <span class="badge">pose <b id="pose">–</b></span>
    <span class="badge">coverage <b id="cov">–</b></span>
    <span class="badge" id="conn">connecting…</span>
    <span class="badge" id="status">–</span>
    <span class="ctl">
      <button class="btn" id="btnPause">⏸ pause</button>
      <button class="btn" id="btnReset">⟳ reset</button>
      <button class="btn danger" id="btnStop">⏹ stop</button>
    </span>
  </div>
  <div class="chips" id="chips">
    <span class="lbl">panels:</span>
    <button class="chip" data-p="sim">🧊 simulation</button>
    <button class="chip" data-p="map">🗺 slam map</button>
    <button class="chip" data-p="lidar">📡 lidar</button>
    <button class="chip" data-p="cloud">☁ point cloud</button>
    <button class="chip" data-p="vel">📈 velocity</button>
    <button class="chip" data-p="hc">📉 height/coverage</button>
    <button class="chip" data-p="raw">🧾 raw telemetry</button>
  </div>
</header>
<div class="wrap">
  <div class="card main" data-panel="sim"><h2>Simulation — 3D (drag to orbit · scroll to zoom)<span class="x" data-close="sim">✕</span></h2>
    <div id="view3d"><div class="msg" id="v3msg">loading 3D…</div></div></div>
  <div class="card" data-panel="map"><h2>SLAM occupancy map<span class="x" data-close="map">✕</span></h2><canvas id="map" width="320" height="320"></canvas></div>
  <div class="card" data-panel="lidar"><h2>Lidar (top-down)<span class="x" data-close="lidar">✕</span></h2><canvas id="lidar" width="320" height="320"></canvas></div>
  <div class="card" data-panel="cloud"><h2>Point cloud (top-down)<span class="x" data-close="cloud">✕</span></h2><canvas id="cloud" width="320" height="320"></canvas></div>
  <div class="card" data-panel="vel"><h2>Base velocity<span class="x" data-close="vel">✕</span></h2><canvas id="vel" width="320" height="150"></canvas></div>
  <div class="card" data-panel="hc"><h2>Base height / coverage<span class="x" data-close="hc">✕</span></h2><canvas id="hc" width="320" height="150"></canvas></div>
  <div class="card" data-panel="raw"><h2>Raw telemetry<span class="x" data-close="raw">✕</span></h2><pre id="raw">–</pre></div>
</div>

<!-- CORE dashboard: telemetry + 2D panels + controls. Runs ALWAYS, independent
     of three.js so a CDN/3D failure never stops data capture. -->
<script>
(function(){
const $=id=>document.getElementById(id);
const hist={vx:[],vy:[],vyaw:[],h:[],cov:[]};
let tPrev=performance.now(),stepPrev=0;
const DEF={sim:1,map:1,lidar:1,cloud:1,vel:1,hc:1,raw:0};
window.vis=Object.assign({},DEF,JSON.parse(localStorage.getItem('domo_panels')||'{}'));
const vis=window.vis;
function applyVis(){for(const ch of document.querySelectorAll('.chip')){const p=ch.dataset.p,on=!!vis[p];
  ch.classList.toggle('on',on);const c=document.querySelector('[data-panel="'+p+'"]');if(c)c.classList.toggle('hidden',!on);}
  if(window.__onResize)window.__onResize();}
function toggle(p){vis[p]=!vis[p];localStorage.setItem('domo_panels',JSON.stringify(vis));applyVis();}
for(const ch of document.querySelectorAll('.chip'))ch.onclick=()=>toggle(ch.dataset.p);
for(const x of document.querySelectorAll('.x'))x.onclick=()=>toggle(x.dataset.close);
function cmd(c){fetch('/cmd?c='+c,{cache:'no-store'}).catch(()=>{});}
$('btnReset').onclick=()=>cmd('reset');$('btnStop').onclick=()=>{if(confirm('Stop the simulation?'))cmd('stop');};
$('btnPause').onclick=()=>cmd('pause');
applyVis();

function line(cv,series,colors,ymin,ymax){const c=cv.getContext('2d'),W=cv.width,H=cv.height;c.clearRect(0,0,W,H);
  c.strokeStyle='#30363d';c.beginPath();c.moveTo(0,H/2);c.lineTo(W,H/2);c.stroke();
  series.forEach((s,i)=>{const d=hist[s];if(!d.length)return;c.strokeStyle=colors[i];c.lineWidth=1.5;c.beginPath();
    const n=Math.min(d.length,W),off=d.length-n;for(let k=0;k<n;k++){const v=d[off+k],y=H-(v-ymin)/(ymax-ymin)*H;k?c.lineTo(k/n*W,y):c.moveTo(0,y);}c.stroke();});}
function drawMap(rows){const cv=$('map'),c=cv.getContext('2d'),W=cv.width,H=cv.height;c.clearRect(0,0,W,H);
  if(!rows||!rows.length)return;const R=rows.length,C=rows[0].length,s=Math.min(W/C,H/R);
  for(let r=0;r<R;r++)for(let k=0;k<C;k++){const ch=rows[r][k];if(ch==='#')c.fillStyle='#e6edf3';else if(ch==='.')c.fillStyle='#1f2d3d';else continue;c.fillRect(k*s,r*s,s+.5,s+.5);}}
function scatter(id,pts,b){const cv=$(id),c=cv.getContext('2d'),W=cv.width,H=cv.height;c.clearRect(0,0,W,H);
  if(!pts||!pts.length)return;const[xa,xb,ya,yb]=b;for(const p of pts){const x=(p[0]-xa)/(xb-xa)*W,y=H-(p[1]-ya)/(yb-ya)*H,z=p[2]||0;c.fillStyle='hsl('+(200-z*120)+',70%,60%)';c.fillRect(x,y,2,2);}}
function drawLidar(sec){const cv=$('lidar'),c=cv.getContext('2d'),W=cv.width,H=cv.height,cx=W/2,cy=H/2;c.clearRect(0,0,W,H);
  if(!sec)return;const R=Math.min(W,H)/2-6,mx=Math.max(...sec,1);c.strokeStyle='#30363d';c.beginPath();c.arc(cx,cy,R,0,7);c.stroke();
  c.fillStyle='#58a6ff';for(let i=0;i<sec.length;i++){const a=i/sec.length*2*Math.PI,r=Math.min(sec[i]/mx,1)*R;c.fillRect(cx+r*Math.cos(a)-1.5,cy+r*Math.sin(a)-1.5,3,3);}c.fillStyle='#f85149';c.fillRect(cx-2,cy-2,4,4);}

async function tick(){
  let s;
  try{ s=await(await fetch('/state',{cache:'no-store'})).json(); }
  catch(e){ $('conn').textContent='○ no server';$('conn').style.color='#f85149'; return; }
  $('conn').textContent='● live';$('conn').style.color='#3fb950';
  const st=s.status||'running';$('status').textContent=st;
  $('status').style.color=st==='paused'?'#d29922':st==='stopped'?'#f85149':'#3fb950';
  $('btnPause').textContent=st==='paused'?'▶ resume':'⏸ pause';
  $('t').textContent=s.t??'–';
  if(s.pose)$('pose').textContent=s.pose[0].toFixed(1)+', '+s.pose[1].toFixed(1);
  if(s.coverage!=null)$('cov').textContent=s.coverage.toFixed(1)+'%';
  const now=performance.now();
  if(s.t!=null){const hz=(s.t-stepPrev)/((now-tPrev)/1000);if(isFinite(hz)&&hz>0)$('fps').textContent=hz.toFixed(0);stepPrev=s.t;tPrev=now;}
  if(s.vel){hist.vx.push(s.vel[0]);hist.vy.push(s.vel[1]);hist.vyaw.push(s.vel[2]);}
  if(s.height!=null)hist.h.push(s.height);if(s.coverage!=null)hist.cov.push(s.coverage);
  for(const k in hist)if(hist[k].length>600)hist[k].shift();
  if(vis.map)drawMap(s.map);if(vis.lidar)drawLidar(s.lidar);
  if(vis.cloud&&s.cloud&&s.bounds)scatter('cloud',s.cloud,s.bounds);
  if(vis.vel)line($('vel'),['vx','vy','vyaw'],['#58a6ff','#3fb950','#d29922'],-1.5,1.5);
  if(vis.hc)line($('hc'),['h','cov'],['#bc8cff','#f85149'],0,Math.max(...hist.cov,1));
  if(vis.raw)$('raw').textContent=JSON.stringify(s,null,1);
  if(window.__update3D)window.__update3D(s);
  if(!window.__sceneLoaded&&s._has_scene&&window.__loadScene)window.__loadScene();
}
setInterval(tick,150);tick();
})();
</script>

<!-- 3D viewer: fully isolated. Dynamic imports in try/catch so a three.js/CDN
     failure only affects THIS panel, never the telemetry above. -->
<script type="module">
(async ()=>{
  const host=document.getElementById('view3d'),msg=document.getElementById('v3msg');
  let THREE,OrbitControls,ColladaLoader,URDFLoader;
  try{
    THREE=await import('three');
    OrbitControls=(await import('three/addons/controls/OrbitControls.js')).OrbitControls;
    ColladaLoader=(await import('three/addons/loaders/ColladaLoader.js')).ColladaLoader;
    URDFLoader=(await import('urdf-loader')).default;
  }catch(e){
    msg.textContent='3D viewer unavailable — three.js failed to load ('+e+'). '
      +'The rest of the dashboard still works; check the browser console / internet for the CDN.';
    return;
  }
  THREE.Object3D.DEFAULT_UP.set(0,0,1);
  const renderer=new THREE.WebGLRenderer({antialias:true});
  renderer.setPixelRatio(devicePixelRatio);renderer.shadowMap.enabled=true;host.appendChild(renderer.domElement);
  const scene=new THREE.Scene();scene.background=new THREE.Color(0x0d1117);
  const camera=new THREE.PerspectiveCamera(50,1,0.05,200);camera.position.set(6,-6,4);
  const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(2,2,0.3);
  scene.add(new THREE.AmbientLight(0xffffff,0.55));
  const dir=new THREE.DirectionalLight(0xffffff,1.1);dir.position.set(6,-4,10);dir.castShadow=true;
  dir.shadow.mapSize.set(1024,1024);const d=14;Object.assign(dir.shadow.camera,{left:-d,right:d,top:d,bottom:-d,near:1,far:60});scene.add(dir);
  const grid=new THREE.GridHelper(40,40,0x30363d,0x21262d);grid.rotation.x=Math.PI/2;scene.add(grid);
  let robot=null,robotJoints=null;
  function onResize(){const r=host.getBoundingClientRect(),w=Math.max(r.width,10),h=Math.max(r.height,10);
    renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();}
  window.__onResize=onResize;onResize();window.addEventListener('resize',onResize);
  (function animate(){requestAnimationFrame(animate);controls.update();if(window.vis&&window.vis.sim)renderer.render(scene,camera);})();
  function boxMesh(b){const g=new THREE.BoxGeometry(b.size[0],b.size[1],b.size[2]);
    const m=new THREE.MeshStandardMaterial({color:b.color||0x8b949e,roughness:0.9});const mesh=new THREE.Mesh(g,m);
    mesh.castShadow=mesh.receiveShadow=true;mesh.position.set(b.pos[0],b.pos[1],b.pos[2]);
    if(b.quat)mesh.quaternion.set(b.quat[1],b.quat[2],b.quat[3],b.quat[0]);return mesh;}
  window.__loadScene=async ()=>{
    if(window.__sceneLoaded)return;let sc;
    try{const r=await fetch('/scene',{cache:'no-store'});if(r.status!==200)return;sc=await r.json();}catch(e){return;}
    if(!sc)return;window.__sceneLoaded=true;msg.textContent='';
    (sc.boxes||[]).forEach(b=>scene.add(boxMesh(b)));
    const g=new THREE.Mesh(new THREE.PlaneGeometry(80,80),new THREE.MeshStandardMaterial({color:0x161b22,roughness:1}));g.receiveShadow=true;scene.add(g);
    if(sc.robot){msg.textContent='loading robot…';
      const loader=new URDFLoader();
      loader.loadMeshCb=(path,mgr,done)=>{new ColladaLoader(mgr).load(path,c=>done(c.scene),undefined,()=>done(null));};
      loader.load(sc.robot.urdf,rob=>{rob.traverse(o=>{if(o.isMesh){o.castShadow=true;o.receiveShadow=true;}});
        robot=rob;robotJoints=sc.robot.dof_names||[];scene.add(rob);msg.textContent='';},
        undefined,err=>{msg.textContent='robot mesh load failed ('+err+')';});
    }
  };
  window.__update3D=(s)=>{
    if(!robot||!s.base)return;
    robot.position.set(s.base[0],s.base[1],s.base[2]);
    robot.quaternion.set(s.base[4],s.base[5],s.base[6],s.base[3]);      // wxyz->xyzw
    if(s.dof&&robotJoints)for(let i=0;i<robotJoints.length&&i<s.dof.length;i++){
      try{robot.setJointValue(robotJoints[i],s.dof[i]);}catch(e){}}
  };
})();
</script>
</body></html>"""


if __name__ == "__main__":
    main()
