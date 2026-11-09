"""Starting a `concordance` command in the background from the web app.

Each runs in its own transient systemd unit, outside this service's cgroup,
so restarting the web service (every deploy) can't kill it mid-run. If
systemd-run isn't available the work stays queued in the database for the
next run of that command.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]


def start(command: str, schema: str) -> bool:
    """Start `concordance <command> --schema <schema>`; True if launched."""
    unit = f"concordance-{command}-{int(time.time())}"
    try:
        subprocess.run(["systemd-run", "--user", "--collect", f"--unit={unit}", f"--working-directory={REPO}",
                        "--", sys.executable, "-m", "concordance", command, "--schema", schema],
                       check=True, capture_output=True, timeout=15)
        return True
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("couldn't start %s (%s); the work stays queued", command, exc)
        return False
