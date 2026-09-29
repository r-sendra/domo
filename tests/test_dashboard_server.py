"""
Engine-free tests for domo.dashboard: the TelemetryHub, the HTTP routes of a
real (ephemeral-port) server, the DashboardClient round-trip, and the
placement rules the module relies on (HTML constant before the __main__
guard; no PIL/numpy needed unless a frame is pushed).
"""

import json
import os
import threading
import time
import urllib.error
import urllib.request

import pytest

from domo import dashboard
from domo.dashboard import DashboardClient, TelemetryHub, make_server, parse_goto

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def server():
    """A live DashboardServer on an ephemeral port; yields (hub, base_url)."""
    hub = TelemetryHub()
    srv = make_server(0, hub, "test title")
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield hub, f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


def _get(base, path):
    return urllib.request.urlopen(base + path, timeout=2)


def _post(base, path, body: bytes, ctype="application/json"):
    req = urllib.request.Request(base + path, data=body, method="POST",
                                 headers={"Content-Type": ctype})
    return urllib.request.urlopen(req, timeout=2)


def _wait_for(pred, timeout=3.0, period=0.02):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(period)
    return False


# ---------------------------------------------------------------------------
# TelemetryHub
# ---------------------------------------------------------------------------

def test_hub_state_carries_bookkeeping_keys():
    hub = TelemetryHub()
    s = hub.get_state()
    assert s["t"] == 0 and s["_frame_id"] == 0 and s["_has_scene"] is False
    assert isinstance(s["_uptime"], float)


def test_hub_publish_replaces_snapshot_and_counts_frames():
    hub = TelemetryHub()
    hub.publish({"t": 3})
    assert hub.get_state()["t"] == 3
    hub.publish({"t": 4}, frame=b"jpeg-bytes")
    assert hub.get_state()["_frame_id"] == 1
    assert hub.get_frame() == (b"jpeg-bytes", 1)
    hub.publish({"t": 5})                       # no frame → id unchanged
    assert hub.get_frame() == (b"jpeg-bytes", 1)


def test_hub_command_is_read_and_clear():
    hub = TelemetryHub()
    assert hub.pop_command() is None
    hub.set_command("pause")
    assert hub.pop_command() == "pause"
    assert hub.pop_command() is None


def test_hub_scene_flags_state():
    hub = TelemetryHub()
    hub.set_scene({"boxes": []}, {"robot": "/nowhere"})
    assert hub.get_state()["_has_scene"] is True
    assert hub.get_scene() == {"boxes": []}


def test_resolve_asset_guards_traversal(tmp_path):
    root = tmp_path / "robot"
    (root / "urdf").mkdir(parents=True)
    (root / "urdf" / "go2.urdf").write_text("<robot/>")
    (tmp_path / "secret.txt").write_text("nope")
    hub = TelemetryHub()
    hub.set_scene({}, {"robot": str(root)})
    assert hub.resolve_asset("robot", "urdf/go2.urdf") == str(root / "urdf" / "go2.urdf")
    assert hub.resolve_asset("robot", "../secret.txt") is None
    assert hub.resolve_asset("robot", "urdf/missing.dae") is None
    assert hub.resolve_asset("other", "urdf/go2.urdf") is None
    assert hub.resolve_asset("robot", "") is None          # the root dir itself is not a file


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------

def test_index_serves_titled_page(server):
    _, base = server
    r = _get(base, "/")
    body = r.read().decode()
    assert r.status == 200
    assert "<title>test title</title>" in body
    assert "/state" in body and "/cmd?c=" in body        # the page polls + controls
    assert _get(base, "/index.html").status == 200
    assert r.headers["Cache-Control"] == "no-store"


def test_state_scene_frame_routes(server):
    hub, base = server
    assert json.loads(_get(base, "/state").read())["t"] == 0
    assert _get(base, "/scene").status == 204
    assert _get(base, "/frame.jpg").status == 204
    hub.set_scene({"boxes": [{"size": [1, 1, 1], "pos": [0, 0, 0]}]})
    hub.set_frame(b"\xff\xd8fake")
    assert json.loads(_get(base, "/scene").read())["boxes"][0]["size"] == [1, 1, 1]
    r = _get(base, "/frame.jpg")
    assert r.status == 200 and r.read() == b"\xff\xd8fake"
    assert r.headers["Content-Type"] == "image/jpeg"


def test_cmd_route_validates_and_relays(server):
    hub, base = server
    for cmd in ("pause", "reset", "stop"):
        assert _get(base, f"/cmd?c={cmd}").read() == b"ok"
        assert json.loads(_get(base, "/cmd-poll").read()) == {"cmd": cmd}
    assert json.loads(_get(base, "/cmd-poll").read()) == {"cmd": None}
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base, "/cmd?c=explode")
    assert e.value.code == 400
    assert hub.pop_command() is None


def test_parse_goto_reads_floats_and_rejects_anything_else():
    """The goal payload is the one command that carries data — strictly parsed."""
    assert parse_goto("goto:2,1") == (2.0, 1.0)
    assert parse_goto("goto:-1.25,-0.50") == (-1.25, -0.5)
    assert parse_goto("goto:0.00,3.75") == (0.0, 3.75)
    for bad in ("goto:", "goto:1", "goto:a,b", "goto:1,2,3", "goto:1,", "goto:,1",
                "goto:1 2", "goto:1,2 ", " goto:1,2", "gotox:1,2", "pause",
                "goto:1,2;rm", None, ""):
        assert parse_goto(bad) is None


def test_cmd_route_relays_a_goto_goal(server):
    """Browser click → /cmd → /cmd-poll, verbatim, including negatives."""
    hub, base = server
    for payload, xy in (("goto:2,1", (2.0, 1.0)),
                        ("goto:-1.25,-0.50", (-1.25, -0.5)),
                        ("goto:0.00,4.50", (0.0, 4.5))):
        assert _get(base, f"/cmd?c={payload}").read() == b"ok"
        got = json.loads(_get(base, "/cmd-poll").read())["cmd"]
        assert got == payload and parse_goto(got) == xy
    assert json.loads(_get(base, "/cmd-poll").read()) == {"cmd": None}
    assert hub.pop_command() is None


def test_cmd_route_rejects_malformed_goals_without_queueing_them(server):
    """A malformed payload is a 400 and never reaches the sim."""
    hub, base = server
    for bad in ("goto:", "goto:1", "goto:a,b", "goto:1,2,3", "goto:1.2.3,4", "goto"):
        with pytest.raises(urllib.error.HTTPError) as e:
            _get(base, f"/cmd?c={bad}")
        assert e.value.code == 400
        assert e.value.read() == b"bad command"
        assert hub.pop_command() is None
        assert json.loads(_get(base, "/cmd-poll").read()) == {"cmd": None}


def test_cmd_route_relays_clear(server):
    """`clear` cancels a goal; it is a bare command like pause/reset/stop."""
    _, base = server
    assert "clear" in dashboard.COMMANDS
    assert _get(base, "/cmd?c=clear").read() == b"ok"
    assert json.loads(_get(base, "/cmd-poll").read()) == {"cmd": "clear"}


def test_unknown_routes_404(server):
    _, base = server
    for path in ("/nope", "/assets/", "/assets/robot/x.dae"):
        with pytest.raises(urllib.error.HTTPError) as e:
            _get(base, path)
        assert e.value.code == 404
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base, "/nope", b"{}")
    assert e.value.code == 404


def test_post_routes_feed_the_hub(server):
    hub, base = server
    assert _post(base, "/ingest", json.dumps({"t": 42, "status": "paused"}).encode()).status == 200
    assert hub.get_state()["t"] == 42
    _post(base, "/scene", json.dumps({"scene": {"boxes": []}, "roots": {}}).encode())
    assert hub.get_scene() == {"boxes": []}
    _post(base, "/frame", b"JPEG", "image/jpeg")
    assert hub.get_frame() == (b"JPEG", 1)


def test_malformed_posts_are_dropped_not_fatal(server):
    hub, base = server
    hub.publish({"t": 7})
    assert _post(base, "/ingest", b"not json").status == 200
    assert _post(base, "/ingest", b"[1, 2]").status == 200      # not a dict
    assert hub.get_state()["t"] == 7
    assert _post(base, "/scene", b"[]").status == 200           # no .get → dropped
    assert hub.get_scene() is None


def test_assets_route_serves_registered_root_only(server, tmp_path):
    hub, base = server
    root = tmp_path / "go2"
    (root / "urdf").mkdir(parents=True)
    (root / "urdf" / "go2.urdf").write_text("<robot name='go2'/>")
    (tmp_path / "outside.txt").write_text("secret")
    hub.set_scene({}, {"robot": str(root)})
    r = _get(base, "/assets/robot/urdf/go2.urdf")
    assert r.status == 200 and b"go2" in r.read()
    assert r.headers["Content-Type"] == "application/xml"
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base, "/assets/robot/../outside.txt")
    assert e.value.code == 404


# ---------------------------------------------------------------------------
# DashboardClient (sim side) ↔ server
# ---------------------------------------------------------------------------

def test_client_round_trip(server):
    hub, base = server
    client = DashboardClient(base, post_hz=100).start()
    try:
        client.set_scene({"boxes": []}, {"robot": "/nowhere"})
        client.publish({"t": 1})
        assert _wait_for(lambda: hub.get_state()["t"] == 1)
        assert _wait_for(lambda: hub.get_scene() == {"boxes": []})
        # Only the LATEST snapshot survives between posts.
        client.publish({"t": 2})
        client.publish({"t": 3})
        assert _wait_for(lambda: hub.get_state()["t"] == 3)
        # Commands flow back: browser → hub → client.poll_command (once).
        hub.set_command("stop")
        assert _wait_for(lambda: client.poll_command() == "stop")
        assert client.poll_command() is None
    finally:
        client.stop()


def test_client_survives_missing_server():
    client = DashboardClient("http://127.0.0.1:9", post_hz=100).start()   # discard port
    try:
        client.publish({"t": 1})
        time.sleep(0.2)
        assert client._thread.is_alive()          # network errors never kill the thread
        assert client.poll_command() is None
    finally:
        client.stop()


# ---------------------------------------------------------------------------
# Module-level invariants
# ---------------------------------------------------------------------------

def test_html_constant_precedes_main_guard():
    """`python -m domo.dashboard` runs top-to-bottom: the page constant must be
    defined before the __main__ guard, which must be the last statement."""
    with open(os.path.join(os.path.dirname(dashboard.__file__), "dashboard.py")) as f:
        src = f.read()
    assert src.index("_DASHBOARD_HTML = r\"\"\"") < src.index('if __name__ == "__main__":')
    assert src.rstrip().endswith("main()")
    assert '"""' not in dashboard._DASHBOARD_HTML     # it lives in an r\"\"\" literal


def test_page_splits_core_and_3d_scripts():
    """Telemetry JS must not sit inside the three.js module, or a CDN failure
    would blank the whole page instead of just the 3D panel."""
    html = dashboard._DASHBOARD_HTML
    core = html.index("<script>\n(function(){")
    module = html.index('<script type="module">')
    assert core < module
    assert "setInterval(tick" in html[core:module]
    assert "import('three')" in html[module:]


def test_click_to_goal_lives_in_the_core_script():
    """The click handler is core interaction: a three.js/CDN failure must not
    take it down with the 3D panel, so it belongs to the plain <script>."""
    html = dashboard._DASHBOARD_HTML
    core = html[html.index("<script>\n(function(){"):html.index('<script type="module">')]
    assert "onclick" in core and "'/cmd?c='" in html
    for piece in ("cmd('goto:'", "toFixed(2)", "pxToWorld", "drawMarks",
                  "cmd('clear')"):
        assert piece in core, piece
    # The goal marker reads the same bounds the cloud is drawn with, and the
    # panel says what a click does.
    assert "cloudBounds=s.bounds" in core
    assert "click to set a goal" in html and 'id="goalClear"' in html


def test_encode_jpeg_from_float_frame():
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    frame = np.random.rand(8, 6, 3) * 255.0        # float → clipped to uint8
    data = dashboard._encode_jpeg(frame)
    assert data[:2] == b"\xff\xd8"                   # JPEG SOI marker
