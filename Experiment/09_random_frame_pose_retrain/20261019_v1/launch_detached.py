"""Start the supervised job without inheriting a tool terminal or its job object."""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if not (HERE / "freeze.json").is_file():
    raise FileNotFoundError("Prepare frozen inputs first")
logs = HERE / "logs"
logs.mkdir(exist_ok=True)
if (logs / "status.json").exists():
    raise FileExistsError("Inspect existing supervision before retrying")
environment = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
with (logs / "launcher.stdout.log").open("x", encoding="utf-8") as stdout:
    with (logs / "launcher.stderr.log").open("x", encoding="utf-8") as stderr:
        process = subprocess.Popen([sys.executable, "-u", str(HERE / "run.py"), "pipeline"],
            cwd=str(HERE.parents[2]), env=environment, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
            close_fds=True, creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_BREAKAWAY_FROM_JOB)
print(json.dumps({"supervisor_pid": process.pid, "detached": True, "status": str(logs / "status.json")}))
