"""Rerun the two failed MobileIE seed-8 LSUI runs once, from scratch, after the matched resume ends.

Author, 2026-11-08: "schedule to rerun after everything is done" (one same-recipe rerun of both failed cells).
The original attempts (lsui_s89 mobileie:fixedG-full/8, diverged at epoch 50; mobileie:learnedSP-full/8, diverged
at epoch 86; persistent non-finite AMP gradients in the MobileIE head) are preserved under
<lane>/_failed_attempts/ and their train logs are renamed *.attempt1.log; liuwir would otherwise resume from the
diverged last.pt. Each cell is retrained exactly once with the frozen matched launcher's children, smoke chain
(default -> amended16 -> relaxed, AMP-skip failures only), recipe, data, seed and MPS. A rerun that fails again is
recorded as failed and NOT retried. No evaluation: test400 stays unscored.

This file lives in a git worktree (the background session could not write to the shared checkout); ROOT points at
the main checkout so the frozen launcher, src/ and outputs/ there are used unchanged.
"""
import importlib.util
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path("/home/heniath/Experiments/learnable-input-uwir")
os.chdir(ROOT)  # the frozen launcher's helper hashes resolve relative paths against the cwd
sys.path.insert(0, str(ROOT / "scripts"))
_spec = importlib.util.spec_from_file_location("run_paper_lsui_matched", ROOT / "scripts/run_paper_lsui_matched.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)  # frozen matched launcher (hash checked by m.check_frozen)
queue, STUDY, LAUNCH = m.queue, m.STUDY, m.LAUNCH
STAGE = "lsui_s89"
RERUNS = (("lane0", "mobileie:fixedG-full", 8), ("lane1", "mobileie:learnedSP-full", 8))
RESUME_RECORD = LAUNCH / "amendment_matched_resume.json"
RECORD = LAUNCH / "amendment_failed_rerun.json"
AUTH = ("Author, 2026-11-08: \"schedule to rerun after everything is done\" (one same-recipe rerun of the two "
        "failed MobileIE seed-8 cells)")


def busy():
    for pattern in ("scripts/resume_paper_lsui_matched.py", "liuwir train", "liuwir smoke", "lsui_liuwir.py"):
        if subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0:
            return True
    return not (RESUME_RECORD.exists() and "ended_utc" in queue.read(RESUME_RECORD))


def wait_for_resume():
    print(f"{m.utc()} waiting for the matched resume to end", flush=True)
    while busy():
        if (STUDY / "STOP").exists():
            raise SystemExit("STOP file present")
        time.sleep(300)
    print(f"{m.utc()} matched resume ended; GPU free", flush=True)


def preserve_attempt(lane, condition, seed, stamp):
    out = STUDY / STAGE / lane
    moved = {}
    for kind, path in (("run", out / "runs" / f"seed_{seed}" / condition),
                       ("smoke", out / "smoke" / f"{condition}_seed{seed}")):
        if path.exists():
            target = out / "_failed_attempts" / f"{kind}_{condition}_seed{seed}__attempt1_diverged_{stamp}"
            target.parent.mkdir(parents=True, exist_ok=True)
            path.rename(target)
            moved[kind] = str(target.relative_to(STUDY))
    log = m.log_path(f"{STAGE}_train_{condition}_seed{seed}")
    if log.exists():
        target = log.with_name(log.stem + ".attempt1.log")
        log.rename(target)
        moved["train_log"] = target.name
    return moved


def rerun(lane, condition, seed, mps, results, lock):
    paths = queue.read(LAUNCH / "protocol.json")["paths"]
    directory = STUDY / STAGE
    common = ["--config", str(directory / f"{lane}.config.json"), "--output-root", str(directory),
              "--data-root", paths["data_root"], "--vgg-weights", paths["vgg_weights"]]
    where = ["--condition", condition, "--seed", str(seed)]
    smoke = directory / lane / "smoke" / f"{condition}_seed{seed}"
    result = {"lane": lane, "run": f"{STAGE} {condition}/{seed}", "started_utc": m.utc(), "smoke_events": []}
    try:
        ok, text = m.call(STAGE, f"{STAGE}_smoke_{condition}_seed{seed}", ["smoke", *common, *where], mps)
        for label, script in (("default_amp_overflow", "scripts/lsui_amended_smoke.py"),
                              ("amended16_amp_overflow", "scripts/lsui_relaxed_smoke.py")):
            if ok:
                break
            if not m.amp_skip_only(text):
                raise RuntimeError("smoke failed for another reason")
            result["smoke_events"].append(f"{m.utc()} {label}; running {script}")
            m.preserve(smoke, label)
            ok, text = m.call(STAGE, f"{STAGE}_{Path(script).stem}_{condition}_seed{seed}", [*common, *where], mps,
                              script=script)
        if not ok:
            raise RuntimeError("relaxed smoke failed")
        queue.status(**{f"{lane}_current": f"{STAGE} {condition}/seed{seed} (rerun)"})
        ok, _ = m.call(STAGE, f"{STAGE}_train_{condition}_seed{seed}", ["train", *common, *where], mps)
        if not ok:
            raise RuntimeError("training failed again (not retried)")
        result["outcome"] = "complete"
    except Exception as exc:
        result.update(outcome="failed", error=str(exc))
    result["ended_utc"] = m.utc()
    with lock:
        results.append(result)
        queue.write(RECORD, {**queue.read(RECORD), "reruns": results})


def main():
    m.check_frozen()
    if RECORD.exists():
        raise FileExistsError(f"{RECORD} exists; this rerun already ran")
    wait_for_resume()
    m.check_frozen()
    stamp = time.strftime("%Y%m%dT%H%M%S")
    preserved = {f"{c}/{s}": preserve_attempt(lane, c, s, stamp) for lane, c, s in RERUNS}
    queue.write(RECORD, {
        "created_utc": m.utc(), "authorization": AUTH,
        "runs": [f"{STAGE} {c}/{s} ({lane})" for lane, c, s in RERUNS],
        "original_failure": "persistent non-finite AMP gradients in the MobileIE head (loss scale 1, 8 reductions); "
                            "fixedG-full/8 at epoch 50, learnedSP-full/8 at epoch 86",
        "preserved_first_attempts": preserved,
        "policy": "one rerun from scratch per cell, same children/smoke chain/recipe/data/seed/MPS; a second failure "
                  "is reported as failed and not retried",
        "evaluation": "not run; test400 unscored until the author approves evaluation",
        "rerun_script": str(Path(__file__).resolve()), "rerun_sha256": queue.sha(Path(__file__)), "reruns": []})
    mps = {"CUDA_MPS_PIPE_DIRECTORY": str(LAUNCH / "mps/pipe"), "CUDA_MPS_LOG_DIRECTORY": str(LAUNCH / "mps/log")}
    for d in mps.values():
        Path(d).mkdir(parents=True, exist_ok=True)
    subprocess.run(["nvidia-cuda-mps-control", "-d"], env=m.env(mps), check=True)
    queue.status(state="running", stage="lsui_failed_rerun", rerun_authorization=AUTH, rerun_started_utc=m.utc())
    results, lock = [], threading.Lock()
    try:
        threads = [threading.Thread(target=rerun, args=(lane, c, s, mps, results, lock)) for lane, c, s in RERUNS]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        subprocess.run(["nvidia-cuda-mps-control"], input=b"quit\n", env=m.env(mps))
        queue.write(RECORD, {**queue.read(RECORD), "reruns": results, "ended_utc": m.utc()})
    m.rl.review()
    prior = queue.read(LAUNCH / "status.json").get("failures") or []
    rerun_names = {f"{STAGE} {c}/{s}" for _, c, s in RERUNS}
    kept = [f for f in prior if not any(n in f for n in rerun_names)]
    failed = [f"{r['lane']}: {r['run']} rerun failed: {r.get('error')}" for r in results if r["outcome"] != "complete"]
    queue.status(state="trained_with_failures" if kept or failed else "trained", stage="lsui_failed_rerun",
                 rerun_ended_utc=m.utc(), failures=[*kept, *failed],
                 rerun_outcomes={r["run"]: r["outcome"] for r in results},
                 note="Training ended. Not evaluated; test400 unscored until the author approves evaluation.")
    print({r["run"]: r["outcome"] for r in results}, flush=True)


if __name__ == "__main__":
    main()
