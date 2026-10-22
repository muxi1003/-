"""Report live process/heartbeat evidence instead of trusting a saved stage alone."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name != "nt":
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return False
        return True
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:
            return False
        raise OSError(error, f"Cannot inspect PID {pid}; do not assume it is dead")
    try:
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return code.value == 259
    finally:
        kernel.CloseHandle(handle)


def observed_state(record: dict, parent_alive: bool, child_alive: bool, age: float | None) -> str:
    if record.get("stage") in ("complete", "failed", "interrupted"):
        return record["stage"]
    if not parent_alive or (record.get("child_pid") and not child_alive):
        return "interrupted"
    if age is None or age > 120:
        return "stale_heartbeat"
    return "running"


def inspect(path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    parent, child = process_alive(record.get("pid")), process_alive(record.get("child_pid"))
    heartbeat = record.get("heartbeat_at")
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(heartbeat)).total_seconds() if heartbeat else None
    return {"observed_state": observed_state(record, parent, child, age),
            "saved_stage": record.get("stage"), "parent_pid": record.get("pid"), "parent_alive": parent,
            "child_pid": record.get("child_pid"), "child_alive": child, "heartbeat_age_seconds": age,
            "note": "Check both heartbeat and live processes. PID may change on recovery; stage alone is not liveness evidence."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("status", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.status), ensure_ascii=False, indent=2))
