"""Backend side of the lease heartbeat and the cross-process prompt inbox (tui_gateway).

Two backends, one chat: the second sender is told "queued" instead of being refused, and the owner (or the
sender's own backend once the old lease has expired) runs the text as an ordinary next turn.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from hermes_cli import active_sessions as AS
from hermes_cli import session_prompt_queue as Q
from hermes_cli.active_sessions import (
    SESSION_NOT_OWNED,
    active_session_registry_snapshot,
    try_acquire_active_session,
)
from tui_gateway import server


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    # The ticker thread must not start in unit tests; we call its ticks by hand.
    monkeypatch.setattr(server, "_start_lease_ticker", lambda: None)
    server._sessions.clear()
    yield home
    server._sessions.clear()


def _session(key="chat-1", **extra):
    return {
        "session_key": key, "history_lock": threading.RLock(), "running": False, "transport": None,
        "profile_home": None, "agent": None, **extra,
    }


def _other_window_holds(key="chat-1", heartbeat=True):
    lease, err = try_acquire_active_session(
        session_id=key, surface="desktop", config={}, heartbeat=heartbeat,
        metadata={"live_session_id": "another-window"})
    assert err is None
    return lease


def test_second_window_is_told_queued_not_refused():
    _other_window_holds()
    mine = _session()
    refusal = server._ensure_active_session_slot("sid-2", mine)
    assert refusal is not None and refusal.reason == SESSION_NOT_OWNED

    reply = server._queue_for_other_owner("rid-1", mine, "also check Rayburn")
    assert reply["result"]["status"] == "queued" and reply["result"]["queued_for_owner"] is True
    assert "another Hermes window" in reply["result"]["message"]
    assert Q.pending_count("chat-1") == 1


def test_a_send_that_cannot_queue_is_still_refused():
    assert server._queue_for_other_owner("rid-1", _session(), ["not", "text"]) is None
    assert server._queue_for_other_owner("rid-1", _session(key=""), "text") is None


def test_owner_delivers_queued_prompts_as_next_turns_in_order(monkeypatch):
    persisted, drained = [], []
    monkeypatch.setattr(server, "_persist_queued_user_row", lambda session, env, kind: persisted.append(env["text"]))
    monkeypatch.setattr(server, "_drain_queued_prompt", lambda rid, sid, session: drained.append(sid) or True)
    owner = _session()
    lease, err = server._claim_active_session_slot("chat-1", live_session_id="sid-1")
    assert err is None and lease.heartbeat is True
    owner["active_session_lease"] = lease

    Q.enqueue_prompt("chat-1", "first")
    Q.enqueue_prompt("chat-1", "second")
    assert server._deliver_inbox_prompts("sid-1", owner, "chat-1", None) == 2

    assert Q.pending_count("chat-1") == 0
    assert owner["queued_prompt"]["text"] == "first\n\nsecond"  # text-only follow-ups merge losslessly, in order
    assert drained == ["sid-1"]  # idle session: the head runs now


def test_a_busy_owner_queues_but_does_not_start_a_second_turn(monkeypatch):
    monkeypatch.setattr(server, "_persist_queued_user_row", lambda *a: None)
    drained = []
    monkeypatch.setattr(server, "_drain_queued_prompt", lambda *a: drained.append(1) or True)
    owner = _session(running=True)
    lease, _ = server._claim_active_session_slot("chat-1", live_session_id="sid-1")
    owner["active_session_lease"] = lease
    Q.enqueue_prompt("chat-1", "after this turn")
    assert server._deliver_inbox_prompts("sid-1", owner, "chat-1", None) == 1
    assert owner["queued_prompt"]["text"] == "after this turn" and drained == []  # the turn's own end drains it


def test_stale_owner_expires_then_the_senders_backend_takes_the_chat_and_runs_the_text(monkeypatch, _home):
    monkeypatch.setattr(server, "_persist_queued_user_row", lambda *a: None)
    drained = []
    monkeypatch.setattr(server, "_drain_queued_prompt", lambda rid, sid, session: drained.append(sid) or True)
    _other_window_holds()
    sender = _session()
    assert server._queue_for_other_owner("rid", sender, "do not lose me") is not None
    # Still held: the sender's tick must not steal the chat or touch the inbox.
    assert server._deliver_inbox_prompts("sid-2", sender, "chat-1", None) == 0
    assert Q.pending_count("chat-1") == 1

    path = _home / "runtime" / "active_sessions.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for e in data["entries"]:
        e["heartbeat_at"] = time.time() - AS.LEASE_EXPIRY_SECONDS - 1
    path.write_text(json.dumps(data), encoding="utf-8")

    assert server._deliver_inbox_prompts("sid-2", sender, "chat-1", None) == 1
    assert sender["queued_prompt"]["text"] == "do not lose me" and drained == ["sid-2"]
    [entry] = active_session_registry_snapshot()
    assert entry["metadata"]["live_session_id"] == "sid-2"


def test_heartbeat_tick_keeps_attached_sessions_and_lets_abandoned_ones_lapse(_home):
    live = _session("chat-live", transport=object())
    gone = _session("chat-gone", transport=server._detached_ws_transport)
    for sid, key, s in (("a", "chat-live", live), ("b", "chat-gone", gone)):
        lease, err = server._claim_active_session_slot(key, live_session_id=sid)
        assert err is None
        s["active_session_lease"] = lease
        server._sessions[sid] = s
    path = _home / "runtime" / "active_sessions.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for e in data["entries"]:
        e["heartbeat_at"] = time.time() - 40
    path.write_text(json.dumps(data), encoding="utf-8")

    server._lease_heartbeat_tick()

    beats = {e["session_id"]: time.time() - e["heartbeat_at"] for e in active_session_registry_snapshot()}
    assert beats["chat-live"] < 5      # refreshed
    assert beats["chat-gone"] >= 39    # client gone and idle: not refreshed, will expire at 60 s


def test_a_session_whose_lease_expired_forgets_it_and_reclaims_next_turn(_home):
    s = _session("chat-1", transport=object())
    lease, _ = server._claim_active_session_slot("chat-1", live_session_id="sid-1")
    s["active_session_lease"] = lease
    server._sessions["sid-1"] = s
    path = _home / "runtime" / "active_sessions.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for e in data["entries"]:
        e["heartbeat_at"] = time.time() - AS.LEASE_EXPIRY_SECONDS - 1
    path.write_text(json.dumps(data), encoding="utf-8")
    # Someone else reads the registry: the stale entry is pruned.
    assert active_session_registry_snapshot() == []

    server._lease_heartbeat_tick()
    assert s.get("active_session_lease") is None
    assert server._ensure_active_session_slot("sid-1", s) is None  # next turn re-claims
    assert s["active_session_lease"] is not None
