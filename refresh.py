"""Cloud Run Job entrypoint: refresh market and news data into TAIEX_DATA."""
import datetime as dt
import json
import os
import subprocess
import sys
import time

CODE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("TAIEX_DATA") or CODE
os.makedirs(DATA, exist_ok=True)
os.chdir(DATA)
STATUS = os.path.join(DATA, "refresh_status.json")

def write_status(**values):
    state = {"updated_at": dt.datetime.now(dt.timezone.utc).isoformat(), **values}
    with open(STATUS, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
        f.flush()

def main():
    steps = ["pipeline", "news"]
    results = []
    write_status(running=True, current="starting", results=[])
    overall = 0
    for step in steps:
        write_status(running=True, current=step, results=results)
        try:
            p = subprocess.run(
                [sys.executable, os.path.join(CODE, step + ".py")],
                cwd=DATA, capture_output=True, text=True, timeout=6 * 3600
            )
            result = {"step": step, "exit_code": p.returncode,
                      "log": (p.stdout + p.stderr)[-4000:]}
        except Exception as e:
            result = {"step": step, "exit_code": 1,
                      "log": f"{type(e).__name__}: {e}"}
        results.append(result)
        if result["exit_code"]:
            overall = result["exit_code"]
    write_status(running=False, current=None, exit_code=overall, results=results)
    return overall

if __name__ == "__main__":
    sys.exit(main())
