"""``GET /api/session-attach`` -- the owner's half of the shared-runtime handshake.

``hermes_cli.shared_session_attach.discover_attach_url`` (the client) reads the lease owner's
``metadata.shared_runtime_url`` from the active-session registry and calls this route to get the owner's
authenticated WebSocket URL, so a second window or terminal attaches to the running chat as a watcher and
never takes (or contests) the lease.

The route answers only when ALL of these hold, otherwise it is a plain 404 that reveals nothing:

* the caller is a loopback address (the route is in the public-path list, so this check IS the gate);
* ``lease_id`` names a live registry entry for ``session_id`` that THIS process owns (pid match), so a
  lease id read from the registry file is useless against any other backend;
* ``profile_home`` is the home that registry belongs to.

The reply echoes the identity (``session_id``, ``lease_id``, ``profile_home``) the client checks, plus a
``websocket_url`` on this process's own loopback origin. Loopback mode hands back the process session token
(the same credential the dashboard's own chat uses); gated mode mints a single-use 30 s ticket instead.
"""

from __future__ import annotations

import ipaddress
import logging
import os
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request

from hermes_cli.active_sessions import active_session_registry_snapshot

_log = logging.getLogger(__name__)
router = APIRouter()


def _is_loopback(host: Optional[str]) -> bool:
    try:
        return bool(host) and ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def owned_lease(session_id: str, lease_id: str, profile_home: str) -> Optional[dict[str, Any]]:
    """The live registry entry for ``lease_id`` when it belongs to ``session_id`` and to this very process."""
    try:
        home = Path(profile_home).resolve()
        entries = active_session_registry_snapshot(home, strict=True)
    except Exception:
        return None
    for entry in entries:
        if (str(entry.get("lease_id") or "") == lease_id and str(entry.get("session_id") or "") == session_id
                and entry.get("pid") == os.getpid()):
            return entry
    return None


@router.get("/api/session-attach")
async def session_attach(request: Request, session_id: str = "", lease_id: str = "", profile_home: str = "") -> dict:
    client = request.client.host if request.client else None
    if not (_is_loopback(client) and session_id and lease_id and profile_home):
        raise HTTPException(status_code=404)
    if owned_lease(session_id, lease_id, profile_home) is None:
        raise HTTPException(status_code=404)
    server = request.scope.get("server") or ()
    port = server[1] if len(server) > 1 else getattr(request.app.state, "bound_port", None)
    if not port:
        raise HTTPException(status_code=404)
    base = f"ws://127.0.0.1:{int(port)}/api/ws"
    if bool(getattr(request.app.state, "auth_required", False)):
        from hermes_cli.dashboard_auth.ws_tickets import mint_ticket
        credential = "ticket=" + quote(mint_ticket(user_id="session-attach", provider="session-attach"), safe="")
    else:
        from hermes_cli.web_server import _SESSION_TOKEN
        credential = "token=" + quote(_SESSION_TOKEN, safe="")
    _log.info("Session %s: loopback watcher attached", session_id)
    return {
        "session_id": session_id, "lease_id": lease_id, "profile_home": str(Path(profile_home).resolve()),
        "websocket_url": f"{base}?{credential}",
    }
