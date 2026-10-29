#!/usr/bin/env python3
"""Tiny dependency-free load test: N concurrent guest users run the core journey (demo -> analysis) and we report timings.

    python scripts/loadtest.py --base http://localhost:8000 --users 8 --rounds 2

Exit code 1 if any journey fails or the slowest analysis exceeds --max-seconds (default 60).
"""
import argparse, json, statistics, threading, time, urllib.request


def call(base, method, path, token=None, body=None):
    req = urllib.request.Request(base + "/api/v1" + path, method=method, data=json.dumps(body).encode() if body is not None else None)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read() or b"null")


def journey(base, out):
    t0 = time.time()
    try:
        tok = call(base, "POST", "/auth/guest")["access_token"]
        ds = call(base, "POST", "/datasets/demo", tok)["dataset"]["id"]
        t1 = time.time()
        a = call(base, "POST", f"/datasets/{ds}/analyses", tok, {"target_column": "churned"})
        while True:
            j = call(base, "GET", f"/jobs/{a['job_id']}", tok)
            if j["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.3)
        if j["status"] != "succeeded":
            raise RuntimeError(j.get("error"))
        out.append({"upload_s": t1 - t0, "analysis_s": time.time() - t1})
    except Exception as e:  # noqa: BLE001
        out.append({"error": str(e)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://localhost:8000"); p.add_argument("--users", type=int, default=5)
    p.add_argument("--rounds", type=int, default=1); p.add_argument("--max-seconds", type=float, default=60)
    a = p.parse_args()
    results = []
    for _ in range(a.rounds):
        threads = [threading.Thread(target=journey, args=(a.base, results)) for _ in range(a.users)]
        [t.start() for t in threads]; [t.join() for t in threads]
    ok = [r for r in results if "error" not in r]
    errs = [r["error"] for r in results if "error" in r]
    if ok:
        an = sorted(r["analysis_s"] for r in ok)
        print(f"journeys ok={len(ok)} failed={len(errs)}  analysis p50={statistics.median(an):.1f}s max={an[-1]:.1f}s")
    for e in errs[:5]:
        print("ERROR:", e)
    raise SystemExit(1 if errs or (ok and max(r["analysis_s"] for r in ok) > a.max_seconds) else 0)


if __name__ == "__main__":
    main()
