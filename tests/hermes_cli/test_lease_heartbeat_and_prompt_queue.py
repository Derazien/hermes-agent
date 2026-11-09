"""Lease heartbeat + expiry, the cross-process prompt inbox, and the watcher attach.

Three promises, tested with the real registry code (no mocks of the lock or the files):

1. A lease that opted in to heartbeats is dropped when its owner stops refreshing it; a refreshed one is kept;
   a lease written without a heartbeat (CLI, gateway, older builds) keeps the old rule and never expires.
2. A send that finds its chat owned by another live backend is queued for that owner, not refused, and the
   text is delivered exactly once, in order, even if a claimer dies halfway.
3. A second window attaches to the owner's runtime as a watcher without taking or disturbing the lease.
"""

import itertools
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest

from hermes_cli import active_sessions as AS
from hermes_cli import session_prompt_queue as Q
from hermes_cli.active_sessions import (
    SESSION_NOT_OWNED,
    active_session_registry_snapshot,
    heartbeat_active_session,
    try_acquire_active_session,
)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


_seq = itertools.count()


def acquire(session_id, *, heartbeat=False, home=None):
    # A distinct live_session_id per call = a distinct writer (same pid), i.e. "another window".
    return try_acquire_active_session(
        session_id=session_id, surface="desktop", config={}, registry_home=home,
        metadata={"live_session_id": f"live-{next(_seq)}"}, heartbeat=heartbeat,
    )


def _age_heartbeat(home, seconds):
    path = home / "runtime" / "active_sessions.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for entry in data["entries"]:
        if "heartbeat_at" in entry:
            entry["heartbeat_at"] = time.time() - seconds
    path.write_text(json.dumps(data), encoding="utf-8")


# ── 1. heartbeat + expiry ────────────────────────────────────────────────────────────────────────────────────

def test_defaults_are_15_and_60_seconds():
    assert AS.LEASE_HEARTBEAT_SECONDS == 15.0
    assert AS.LEASE_EXPIRY_SECONDS == 60.0


def test_opted_in_lease_records_a_heartbeat():
    lease, err = acquire("chat-1", heartbeat=True)
    assert err is None
    [entry] = active_session_registry_snapshot()
    assert entry["heartbeat_at"] > 0


def test_stale_lease_expires_and_the_next_window_gets_the_chat(_isolated_home):
    first, _ = acquire("chat-1", heartbeat=True)
    # Another window is refused while the owner is fresh...
    refused, err = acquire("chat-1", heartbeat=True)
    assert refused is None and err.reason == SESSION_NOT_OWNED
    # ...but 61 s without a heartbeat and the lease is gone.
    _age_heartbeat(_isolated_home, AS.LEASE_EXPIRY_SECONDS + 1)
    second, err = acquire("chat-1", heartbeat=True)
    assert err is None and second is not None
    [entry] = active_session_registry_snapshot()
    assert entry["lease_id"] == second.lease_id


def test_a_refreshed_lease_is_kept(_isolated_home):
    lease, _ = acquire("chat-1", heartbeat=True)
    _age_heartbeat(_isolated_home, AS.LEASE_EXPIRY_SECONDS - 5)  # nearly stale
    assert heartbeat_active_session(lease) is True               # owner beats
    _age_heartbeat(_isolated_home, 0)
    refused, err = acquire("chat-1", heartbeat=True)
    assert refused is None and err.reason == SESSION_NOT_OWNED


def test_a_lease_without_heartbeat_never_expires(_isolated_home):
    lease, _ = acquire("chat-1", heartbeat=False)
    [entry] = active_session_registry_snapshot()
    assert "heartbeat_at" not in entry
    refused, err = acquire("chat-1")
    assert refused is None and err.reason == SESSION_NOT_OWNED
    assert lease.heartbeat is False and heartbeat_active_session(lease) is True  # no-op, not a loss


def test_the_heartbeat_reports_a_lost_lease(_isolated_home):
    lease, _ = acquire("chat-1", heartbeat=True)
    _age_heartbeat(_isolated_home, AS.LEASE_EXPIRY_SECONDS + 1)
    taker, err = acquire("chat-1", heartbeat=True)  # prunes the stale entry, takes the chat
    assert err is None
    assert heartbeat_active_session(lease) is False  # the old owner learns it lost
    assert heartbeat_active_session(taker) is True


def test_transfer_keeps_the_heartbeat(_isolated_home):
    lease, _ = acquire("chat-1", heartbeat=True)
    _age_heartbeat(_isolated_home, 30)
    assert AS.transfer_active_session(lease, session_id="chat-2")
    [entry] = active_session_registry_snapshot()
    assert time.time() - entry["heartbeat_at"] < 5


# ── 2. queue, never refuse ───────────────────────────────────────────────────────────────────────────────────

def test_two_backends_one_chat_second_sender_queues(_isolated_home):
    owner, _ = acquire("chat-1", heartbeat=True)
    second, err = acquire("chat-1", heartbeat=True)
    assert second is None and err.reason == SESSION_NOT_OWNED
    # What prompt.submit now does on that refusal:
    qid = Q.enqueue_prompt("chat-1", "please also check the Rayburn invoice", sender={"surface": "desktop"})
    assert qid and Q.pending_count("chat-1") == 1
    # The owner's ticker claims, queues in its own session, then acknowledges.
    [entry] = Q.claim_prompts("chat-1")
    assert entry.text == "please also check the Rayburn invoice"
    assert Q.pending_count("chat-1") == 0  # claimed entries are not claimable twice
    assert Q.claim_prompts("chat-1") == []
    Q.ack(entry)
    assert not entry.path.exists()


def test_queue_keeps_arrival_order_and_chats_apart():
    for text in ("one", "two", "three"):
        Q.enqueue_prompt("chat-1", text)
    Q.enqueue_prompt("chat-2", "elsewhere")
    assert [e.text for e in Q.claim_prompts("chat-1")] == ["one", "two", "three"]
    assert [e.text for e in Q.claim_prompts("chat-2")] == ["elsewhere"]


def test_a_claim_nobody_acknowledged_returns_to_the_queue(monkeypatch):
    Q.enqueue_prompt("chat-1", "do not lose me")
    [entry] = Q.claim_prompts("chat-1")           # claimer dies here: no ack
    assert Q.claim_prompts("chat-1") == []
    old = time.time() - Q.CLAIM_STALE_SECONDS - 5
    os.utime(entry.path, (old, old))
    [again] = Q.claim_prompts("chat-1")
    assert again.text == "do not lose me" and again.queue_id == entry.queue_id


def test_release_puts_a_claim_back():
    Q.enqueue_prompt("chat-1", "later")
    [entry] = Q.claim_prompts("chat-1")
    Q.release(entry)
    assert Q.pending_count("chat-1") == 1


def test_unqueueable_sends_fall_back_to_a_refusal():
    assert Q.enqueue_prompt("chat-1", "   ") is None
    assert Q.enqueue_prompt("", "text") is None
    assert Q.enqueue_prompt("chat-1", "x" * (Q.MAX_TEXT_CHARS + 1)) is None


def test_two_claimers_never_get_the_same_prompt():
    for i in range(30):
        Q.enqueue_prompt("chat-1", f"m{i}")
    got, lock = [], threading.Lock()

    def claim():
        mine = [e.text for e in Q.claim_prompts("chat-1", limit=30)]
        with lock:
            got.extend(mine)

    threads = [threading.Thread(target=claim) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(got) == sorted(f"m{i}" for i in range(30))


def test_session_ids_cannot_escape_the_queue_folder(_isolated_home):
    Q.enqueue_prompt("../../etc/passwd", "x")
    folder = Q.queue_dir("../../etc/passwd")
    assert folder.parent == _isolated_home / "runtime" / "queued_prompts"


# ── 3. watcher attach (no lease taken) ───────────────────────────────────────────────────────────────────────

class _Owner(ThreadingHTTPServer):
    """A loopback server answering /api/session-attach with the same identity rules as the real route."""


def _serve(lease_id_getter):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            parts = urlsplit(self.path)
            q = {k: v[0] for k, v in parse_qs(parts.query).items()}
            from hermes_cli.web_routers.session_attach import owned_lease
            if parts.path != "/api/session-attach" or owned_lease(q["session_id"], q["lease_id"], q["profile_home"]) is None:
                self.send_response(404)
                self.end_headers()
                return
            port = self.server.server_address[1]
            body = json.dumps({**q, "websocket_url": f"ws://127.0.0.1:{port}/api/ws?token=t"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    srv = _Owner(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_watcher_attaches_without_taking_the_lease(_isolated_home):
    pytest.importorskip("fastapi")
    from hermes_cli.shared_session_attach import discover_attach_url

    srv = _serve(None)
    try:
        port = srv.server_address[1]
        lease, _ = try_acquire_active_session(
            session_id="chat-1", surface="desktop", config={}, heartbeat=True,
            metadata={"live_session_id": "owner", "shared_runtime_url": f"http://127.0.0.1:{port}"})
        before = active_session_registry_snapshot()
        url = discover_attach_url("chat-1")
        assert url == f"ws://127.0.0.1:{port}/api/ws?token=t"
        after = active_session_registry_snapshot()
        assert [e["lease_id"] for e in after] == [e["lease_id"] for e in before] == [lease.lease_id]
    finally:
        srv.shutdown()


def test_attach_route_refuses_a_lease_this_process_does_not_own(_isolated_home):
    pytest.importorskip("fastapi")
    from hermes_cli.web_routers.session_attach import owned_lease

    lease, _ = acquire("chat-1", heartbeat=True)
    assert owned_lease("chat-1", lease.lease_id, str(_isolated_home)) is not None
    assert owned_lease("chat-1", "deadbeef", str(_isolated_home)) is None
    assert owned_lease("chat-2", lease.lease_id, str(_isolated_home)) is None
