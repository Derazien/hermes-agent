"""Cross-process prompt inbox: a send that finds its chat owned by another live backend is QUEUED here,
not refused.

The active-session lease (``active_sessions``) keeps two backends from writing one stored session at the
same time. Refusing the second sender loses its message and interrupts the person. Instead the second
backend drops the prompt into ``<registry home>/runtime/queued_prompts/<session>/`` and tells the sender
"queued". Whichever backend owns the chat next (the current owner once its turn ends, or the sender's own
backend once the old lease has expired) claims the files in arrival order and runs them as ordinary
queued turns.

Delivery is at least once and never lossy: an entry is claimed by renaming it, deleted only after the
owner has durably queued it in its own session (``ack``), and a claim that nobody acknowledged within
``CLAIM_STALE_SECONDS`` goes back to the queue. No lock is held across a turn; each step is one
``os.replace`` or ``unlink``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# A claimed entry that was never acknowledged (the claimer died mid-handoff) returns to the queue after this long.
CLAIM_STALE_SECONDS = 300.0
# An unclaimed entry older than this is dropped with a log line; a day-old "run this next" is no longer wanted.
ENTRY_MAX_AGE_SECONDS = 24 * 3600.0
# Hard cap on text per entry and entries per session, so a runaway sender cannot fill the disk.
MAX_TEXT_CHARS = 200_000
MAX_ENTRIES_PER_SESSION = 200

_SUFFIX = ".json"
_CLAIMED = ".claimed"


@dataclass
class QueuedPrompt:
    queue_id: str
    session_id: str
    text: str
    queued_at: float
    sender: dict[str, Any]
    path: Path  # the claimed file (set by ``claim_prompts``)


def _registry_home(registry_home: str | Path | None) -> Path:
    return Path(registry_home) if registry_home is not None else Path(get_hermes_home())


def queue_dir(session_id: str, registry_home: str | Path | None = None) -> Path:
    """One folder per stored session; the name is a hash so no session id can escape the directory."""
    digest = hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()[:24]
    return _registry_home(registry_home) / "runtime" / "queued_prompts" / digest


def enqueue_prompt(
    session_id: str, text: str, *, sender: Optional[dict[str, Any]] = None,
    registry_home: str | Path | None = None,
) -> Optional[str]:
    """Append ``text`` to the session's inbox. Returns the queue id, or None when it cannot be queued
    (empty, too long, inbox full): the caller then falls back to the old refusal so nothing is silently dropped."""
    key = str(session_id or "")
    if not key or not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS:
        return None
    folder = queue_dir(key, registry_home)
    try:
        folder.mkdir(parents=True, exist_ok=True)
        if sum(1 for p in folder.iterdir() if p.name.endswith(_SUFFIX)) >= MAX_ENTRIES_PER_SESSION:
            return None
        now = time.time()
        queue_id = f"{time.time_ns():020d}-{uuid.uuid4().hex[:8]}"
        payload = {
            "queue_id": queue_id, "session_id": key, "text": text, "queued_at": now,
            "sender": {str(k): v for k, v in (sender or {}).items() if isinstance(k, str)},
        }
        tmp = folder / f".{queue_id}.tmp"
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, folder / f"{queue_id}{_SUFFIX}")
        return queue_id
    except OSError:
        logger.warning("Could not queue a prompt for session %s", key, exc_info=True)
        return None


def pending_count(session_id: str, registry_home: str | Path | None = None) -> int:
    """Entries waiting (unclaimed) for the session. Cheap: one directory listing."""
    folder = queue_dir(session_id, registry_home)
    try:
        return sum(1 for p in folder.iterdir() if p.name.endswith(_SUFFIX))
    except OSError:
        return 0


def _revive_stale_claims(folder: Path, now: float) -> None:
    for p in list(folder.iterdir()):
        if _CLAIMED not in p.name:
            continue
        try:
            if now - p.stat().st_mtime > CLAIM_STALE_SECONDS:
                base = p.name.split(_CLAIMED, 1)[0]
                os.replace(p, folder / base)
        except OSError:
            continue


def claim_prompts(
    session_id: str, *, limit: int = 20, registry_home: str | Path | None = None,
) -> list[QueuedPrompt]:
    """Claim up to ``limit`` waiting entries, oldest first. Each returned entry must be ``ack``ed once queued
    in the owner's session, or ``release``d to go back. Two claimers never get the same entry (rename is atomic)."""
    folder = queue_dir(session_id, registry_home)
    now = time.time()
    claimed: list[QueuedPrompt] = []
    try:
        if not folder.is_dir():
            return claimed
        _revive_stale_claims(folder, now)
        names = sorted(p.name for p in folder.iterdir() if p.name.endswith(_SUFFIX))
    except OSError:
        return claimed
    for name in names:
        if len(claimed) >= limit:
            break
        src = folder / name
        dst = folder / f"{name}{_CLAIMED}.{os.getpid()}.{uuid.uuid4().hex[:8]}"  # unique per claim, even within one process
        try:
            os.replace(src, dst)  # whoever wins the rename owns the entry
        except OSError:
            continue
        with suppress(OSError):
            os.utime(dst, None)  # the stale-claim clock starts at the claim, not at the original write
        try:
            raw = dst.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue  # another claimer won the rename race (some platforms report both renames as ok)
        except OSError:
            release_path = dst
            _revert(release_path)
            continue
        try:
            data = json.loads(raw)
            text = data.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("empty")
            queued_at = float(data.get("queued_at") or now)
        except (OSError, ValueError, TypeError):
            logger.warning("Dropped an unreadable queued prompt for session %s", session_id)
            _unlink(dst)
            continue
        if now - queued_at > ENTRY_MAX_AGE_SECONDS:
            logger.warning("Dropped a queued prompt older than a day for session %s", session_id)
            _unlink(dst)
            continue
        sender = data.get("sender") if isinstance(data.get("sender"), dict) else {}
        claimed.append(QueuedPrompt(
            queue_id=str(data.get("queue_id") or name[: -len(_SUFFIX)]), session_id=str(session_id),
            text=text, queued_at=queued_at, sender=sender, path=dst))
    return claimed


def ack(entry: QueuedPrompt) -> None:
    """The owner has durably queued the prompt: remove it from the inbox."""
    _unlink(entry.path)


def release(entry: QueuedPrompt) -> None:
    """Put a claimed entry back (the owner could not take it after all)."""
    _revert(entry.path)


def _revert(path: Path) -> None:
    try:
        os.replace(path, path.parent / path.name.split(_CLAIMED, 1)[0])
    except OSError:
        logger.debug("Could not release a claimed prompt", exc_info=True)


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass
