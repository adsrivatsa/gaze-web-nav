# gaze-web-nav — Project Handoff

**Purpose of this doc:** everything needed to resume the project in a fresh thread. Covers project context, infrastructure, the finalized data-collection recorder, the data formats, the three helper scripts (full source), the key methodology decisions, and the roadmap for the phases not yet done.

**Status right now:** data-collection tooling is complete and verified end-to-end. Login state regenerated and confirmed working. **The immediate next action is to begin collecting human demonstrations** (run `progress.py` for the next task, record 3 trajectories per task). Nothing about the recorder, the split, or the data format should need to change once collection starts.

---

## 1. Project overview

Gaze-regularized fine-tuning of **UI-TARS-1.5-7B** for **WebArena** web-navigation (shopping site). USC LIRA Lab (PI Prof. Erdem Biyik). Researcher: Danie Craig Kulandai. Mentor: Yutai Zhou. Industry tie: Capital One. Repo: **https://github.com/Danie-Craig/gaze-web-nav** (public).

**Core idea:** add a soft KL term that pulls the model's visual attention toward where a human actually looked while performing the task.

```
L_total = L_action + λ · KL( human_gaze ‖ model_attention )
```
- λ ≈ 0.001
- KL computed on the **last transformer layer only**, mode **"mean_then_kl"**
- Gaze KL applied **only to spatially-grounded actions** (see §7).

**Three-model ablation** (goal: C > B > A):
- **A** = base UI-TARS-1.5-7B, no training
- **B** = behavioral cloning (screenshots + actions, no gaze)
- **C** = gaze-regularized (B's dataset + the gaze KL term)

One collection pass with gaze ON serves both B (uses screenshots+actions) and C (also uses gaze) — do **not** collect twice.

> Note: an old name "CREDIF" appears in some earlier docs. It is **wrong** — do not use it.

---

## 2. Infrastructure

**Alienware Aurora R16 — `liralab-widowx`** (WebArena host + training)
- Ubuntu 22.04, RTX 4070 (12 GB), CUDA 12.2
- SSH: `ssh liralab-widowx@100.106.3.51`
- Tailscale IP `100.106.3.51`, hostname `liralabwidowx-alienware-aurora-r16.tail4d611e.ts.net`
- `/data` = 5.3 TB (put large files here). `/` is ~95% full — **never** write large files to `/`.
- Conda env **`gaze-train`**: torch 2.5.1+cu121, transformers 5.6.0, tokenizers 0.22.2, llamafactory 0.9.6.dev0, peft 0.18.1, bitsandbytes 0.49.2. Eval additionally needs playwright 1.32.1.
  - **Caveat:** WebArena's `requirements.txt` pins `transformers==4.33.2`, which breaks UI-TARS. After installing WebArena deps, restore `transformers==5.6.0 tokenizers==0.22.2 huggingface-hub==1.17.0`.

**Windows 11 laptop** (data collection + GazePoint host)
- Tailscale IP `100.104.0.57`
- Conda env **`gaze-web-nav`** (Python 3.11, Playwright 1.60.0) — this is the correct env for the recorder.
- PowerShell 5.1
- Display **DPR = 1.5** (HiDPI). Browser CSS viewport **1280×585**; device screenshot **1920×878**.
- Repo clone at **`D:\gaze\gaze-web-nav\`**; recorder dir **`D:\gaze\gaze-web-nav\data_collection\`**.

**WebArena Shopping site** (OneStopMarket / Magento Luma)
- URL: `http://liralabwidowx-alienware-aurora-r16.tail4d611e.ts.net:8082`
- Login: `emma.lopez@gmail.com` / `Password.123`
- Start on Alienware: `bash /data/webarena-setup/start_webarena.sh`
- Logged-in vs logged-out tell in the page: top-right shows **"Sign Out"** (logged in) vs **"Sign In" / "Create an Account"** (logged out). The "Welcome to One Stop Market" banner is generic and shown to everyone — it is **not** a login indicator.

**GazePoint GP3V2** eye tracker
- 61 Hz, OpenGaze API over TCP, port **4242**, XML stream.
- Recorder sends `ENABLE_SEND_POG_FIX`, `ENABLE_SEND_CURSOR`, `ENABLE_SEND_DATA`; connects to `127.0.0.1:4242`.
- **Must calibrate** in GazePoint Control before each collection session.

---

## 3. Key paths

**Alienware**
- Model: `/data/gaze-web-nav-training/models/UI-TARS-1.5-7B` (33 GB, Qwen2.5-VL architecture)
- `/data/gaze-web-nav-training/LLaMA-Factory/`, `outputs/bc_baseline/` (old Model-B LoRA — obsolete)
- `/data/webarena-setup/` — `start_webarena.sh`, `webarena/00_vars.sh`, `webarena-tasks/` (`config_files/test.raw.json`, `.auth/shopping_state.json`)
- Suggested recordings destination for backup/preprocessing: `/data/gaze-web-nav-training/recordings/`

**Windows `D:\gaze\gaze-web-nav\`**
- `data_collection/record.py` — the recorder (see §4)
- `data_collection/all_shopping_tasks.json` — **187** shopping tasks (the full set used)
- `data_collection/splits.json`, `data_collection/collection_checklist.csv` — produced by `split_tasks.py`, committed
- `data_collection/split_tasks.py`, `progress.py`, `make_login_state.py` — helper scripts (§5, §8, §9)
- `data_collection/shopping_state.json` — logged-in cookies the recorder loads (regenerate when expired)
- `data_collection/test_gaze.py` — 26-line GazePoint smoke test (prints received packets)
- `training/preprocess.py` (OLD — needs overhaul, §10), `configs/bc_baseline.yaml`
- `evaluation/uitars_agent.py`, `generate_configs.py`, `run_eval.py` (OLD — needs fixes, §10)

> `sampled_shopping_tasks.json` (50 tasks) exists but is **NOT used** — we use all 187.

---

## 4. The recorder — `data_collection/record.py` (FINALIZED, in repo, 839 lines)

A Playwright + asyncio recorder run on the Windows laptop. It is **locked and fully validated**; full source is in the repo. This section documents its behavior and config so preprocessing can be written against it.

**What it captures:** all UI-TARS action types **except `wait()`** (deliberately omitted; a demonstrator never needs to wait). `finished()` is emitted at session end (it prompts for the answer string). Action classification (validated correct across dry-runs and the gaze run):
- `click` — single click → `x, y`
- `left_double` — double click (within 0.35 s window) → `x, y`
- `right_single` — right click (button 2) → `x, y`
- `drag` — mousedown→mouseup with distance > 20 px → start + end coords
- `scroll` — wheel → `x, y, direction, delta_x, delta_y, scrollX, scrollY, is_page`
- `type` — text entry, value flushed on Tab/Enter (Shift+Enter excluded) → `value` + `trigger`
- `hotkey` — modifier combos (Ctrl/Alt/Meta + key) → `key`
- `start` / `navigate` — **non-trainable boundary markers** (URL); preprocess drops them as action labels but uses them to segment trajectories and reset the scroll baseline
- `finished` — session end → `content` (the answer)

**Key mechanisms (all validated):**
- Conflict-free pointer state machine: eager screenshot on `mousedown` (background task), action classified on `mouseup`.
- **Coordinates are in SCREENSHOT pixels.** `page.screenshot()` is captured at device scale; the recorder multiplies `clientX/Y` by `dpr` (1.5) so saved `x,y` index the PNG directly. The screenshot is the **content viewport only** (no browser chrome), `1920×878`.
- **Hotkey pre-press screenshot buffer:** a screenshot is buffered the instant a modifier key goes down (before the combo's effect), so e.g. Ctrl+A's frame shows the BEFORE state.
- **Scroll-baseline reset:** on URL change the baseline is re-read from `window.scrollX/Y`, fixing first-scroll-after-navigate delta under-counting. (UI-TARS scroll is direction-only, so magnitude doesn't affect training anyway, but deltas are now internally consistent.)
- **Screen geometry captured once at startup** into `session_info.json` (needed to map gaze → screenshot; see §6). **Do not move or resize the window mid-recording.**
- **Graceful gaze failure:** if GazePoint can't connect, it prints a clear message and aborts rather than silently collecting gaze-less data.
- **Folder-reuse guard:** if the target folder exists and is non-empty, it asks "overwrite? (y/n)".
- **Keep prompt:** at the end it asks "Keep this recording? (y/n)" — answer `n` to discard a fumbled demo and redo.

**Config (top of file):**
- `GAZE_ENABLED` — set `True` for real collection (currently True).
- `TASK_URL` — the shopping Tailscale URL (`...:8082`).
- `STORAGE_STATE = "./shopping_state.json"` — loads logged-in cookies if present.

**Per-recording output** → `recordings/task{ID}_traj{N}/`:
- `screenshots/` — `0000_start.png`, `0001_click.png`, … (named `NNNN_<type>.png`)
- `actions.json`, `gaze.json`, `session_info.json` (schemas in §5)

**Operational rule:** at the "Task ID" prompt, type the **bare integer only** (e.g. `48`), not `task 48`. The recorder adds the `task` prefix itself, and preprocess matches the id against `all_shopping_tasks.json` by integer.

---

## 5. Output data schemas (from real recorded data)

**`actions.json`** — list of action rows. Two timestamps per row: `timestamp` = the **decision moment** (the anchor for gaze + screenshot), `recorded_at` = when the row was written.

```json
{
  "step": 1,
  "type": "click",
  "timestamp": 1782176380.8333788,
  "recorded_at": 1782176381.2669926,
  "screenshot": "0001_click.png",
  "x": 1498, "y": 123
}
```
Field sets by type: `start`/`navigate` → `url`; `click`/`left_double`/`right_single` → `x,y`; `drag` → start+end coords; `scroll` → `x,y,direction,delta_x,delta_y,scrollX,scrollY,is_page`; `type` → `value,trigger`; `hotkey` → `key`; `finished` → `content`.

**`gaze.json`** — list of raw GazePoint samples (or `[]` if gaze disabled). Each sample stores host `time.time()` as `t` (alignable with action `timestamp`) and the verbatim OpenGaze line:

```json
{"t": 1782176354.068,
 "raw": "<REC FPOGX=\"0.40395\" FPOGY=\"0.57100\" FPOGS=\"543.73004\" FPOGD=\"0.18134\" FPOGID=\"1178\" FPOGV=\"1\" CX=\"0.71354\" CY=\"0.97037\" CS=\"0\" />"}
```
Relevant fields in `raw`: `FPOGX`,`FPOGY` = fixation point as fraction [0,1] of the **physical screen**; `FPOGV` = validity flag (1 = valid fixation, 0 = blink/saccade/lost); `FPOGS` = device timestamp; `CX`,`CY` = cursor position. **Filtering rule for preprocess:** keep a sample only if `FPOGV == 1` **and** its mapped coordinates fall inside the screenshot (i.e. `FPOGX,FPOGY` in [0,1] after the §6 mapping). This drops blinks/saccades and off-screen glances (e.g. eyes on the keyboard while typing, which show up as `FPOGY > 1`).

**`session_info.json`** — captured once at startup:

```json
{
  "task_id": "9999", "trajectory": "1", "timestamp": "20260622_175913",
  "gaze_enabled": true,
  "task_url": "http://liralabwidowx-alienware-aurora-r16.tail4d611e.ts.net:8082",
  "device_pixel_ratio": 1.5,
  "inner_width": 1280, "inner_height": 585,
  "outer_width": 1280, "outer_height": 672,
  "screen_width": 1280, "screen_height": 720,
  "window_screen_x": 0, "window_screen_y": 0,
  "screenshot_size": [1920, 878],
  "coordinate_space": "screenshot_pixels (device scale); x,y index the PNG directly"
}
```

---

## 6. Coordinate systems (critical for preprocess)

**Action coordinates** are already in screenshot pixels — they index the PNG directly. No transform needed for the action labels themselves *except* the UI-TARS coordinate convention below.

**UI-TARS-1.5 expects ABSOLUTE PIXEL coordinates** (it is Qwen2.5-VL-based), **not** 0–1000 normalized coordinates. The OLD `preprocess.py` normalizes to 0–1000 — that is **wrong** and is a pending fix (§10). Use ByteDance's `smart_resize` to map raw screenshot pixels to the coordinates the model's image processor actually sees.

**Gaze → screenshot-pixel mapping.** `FPOGX/FPOGY` are fractions of the full physical screen; the screenshot is only the content viewport. Using the `session_info` geometry (all CSS px except dpr):
```
dpr        = device_pixel_ratio
W_screen   = screen_width  * dpr          # e.g. 1280*1.5 = 1920
H_screen   = screen_height * dpr          # e.g.  720*1.5 = 1080
gx = FPOGX * W_screen                      # gaze in device px
gy = FPOGY * H_screen
chrome_x   = (outer_width  - inner_width)  # ~0 here (no side chrome)
chrome_y   = (outer_height - inner_height) # browser toolbar height, e.g. 672-585 = 87
cx0 = (window_screen_x + chrome_x) * dpr   # content viewport top-left, device px
cy0 = (window_screen_y + chrome_y) * dpr
sx = gx - cx0                              # screenshot pixel coords
sy = gy - cy0
# valid iff 0 <= sx < screenshot_w and 0 <= sy < screenshot_h
```
Check with the captured numbers: content height device = 585*1.5 = 877.5 ≈ screenshot height 878; width 1280*1.5 = 1920 = screenshot width 1920. ✔
This assumes a maximized Chrome window (all chrome on top); that's why the window must not move/resize mid-recording, and why geometry is captured once. These fields are unrecoverable after the fact, which is the whole reason they're saved.

---

## 7. Gaze validation result (recorder is proven)

One ~80 s recording with gaze ON (`task9999_traj1`) produced:
- **4843 gaze points at 60.9 Hz** — the tracker's full 61 Hz with essentially zero dropped samples. The background reader keeps up.
- Correct structure (`{"t","raw"}` with `FPOGX/FPOGY/FPOGV`).
- **Timestamps align with actions** (shared host clock): every spatially-grounded action had a full ~12-sample window in the 200 ms before its decision moment, with enough usable samples to build a gaze map.
- Quality: **72% valid fixations / 69% usable** (valid AND on-screen). The ~28% invalid is normal saccades + blinks + the keyboard glance while typing. Healthy and expected.
- `session_info` had `gaze_enabled: true` plus full geometry. Scroll deltas internally consistent on both pages.

Two benign quirks to know: gaze stream *starts* a few seconds before step 0 (thread connects during setup — pre-roll, unused) and *ends* when the browser closes, before the `finished` step is typed at the terminal (`finished` is non-spatial and needs no gaze).

---

## 8. Dataset split (DECIDED, committed)

**Decision:** use all **187** tasks. Split **by task** (never by trajectory/step), **70/15/15**, fixed **seed 42** → **131 train / 28 val / 28 test**. The 15 held-out evaluation task IDs are **forced into test** and never appear in train/val.

**15 forced eval IDs:** `48, 49, 146, 147, 233, 240, 261, 351, 352, 436, 510, 518, 521, 691, 796`

**Collection plan:** collect **3 human demos each for train + val only** = 159 tasks × 3 = **477 trajectories** (the real labor). **Test = automated live WebArena eval, no demos.** Knob if labor is too high: shift to e.g. `--train 0.60 --val 0.15` (test → 25%, fewer demo tasks); the 15 eval IDs stay forced in test regardless. (2 demos/task instead of 3 is the other lever.)

`splits.json` is committed and is the frozen source of truth. The verified real run printed `train: 131 (70.1%) / val: 28 (15.0%) / test: 28 (15.0%)`.

### `split_tasks.py` (full source)
```python
#!/usr/bin/env python3
"""
Split the WebArena shopping tasks into train / val / test BY TASK.

Why "by task": a task and all of its human-demo trajectories must live in the
SAME split. Splitting by trajectory or by step would leak a task the model has
practiced into validation/test and inflate the numbers. So we assign whole
task_ids to splits; collection then records N demos per train/val task.

Guarantees
----------
- Split is by task_id, never by trajectory/step  -> no task leaks across splits.
- Fixed --seed  -> fully reproducible (sort-then-shuffle, independent of file order).
- The 15 held-out evaluation task_ids are FORCED into test and NEVER appear in
  train or val.
- train + val  = the tasks you collect human demos for (--demos each).
  test          = evaluated live by automated WebArena (no demos collected).

Outputs (to --out-dir, default current dir)
-------------------------------------------
  splits.json               canonical manifest: {"train":[...], "val":[...], "test":[...], + metadata}
  collection_checklist.csv  task_id, split, demos_needed, intent   (train+val only)

COMMIT splits.json to the repo. It freezes the split forever; re-running with the
same seed reproduces it, but the committed file is the source of truth.

Usage
-----
  python split_tasks.py                         # 70/15/15, seed 42, 3 demos/task
  python split_tasks.py --train 0.60 --val 0.15 # bigger test => fewer demo tasks
  python split_tasks.py --tasks all_shopping_tasks.json --out-dir ../data
"""
import argparse
import csv
import json
import random
from pathlib import Path

# The 15 evaluation task_ids that MUST land in test (and never in train/val).
EVAL_TASK_IDS = [48, 49, 146, 147, 233, 240, 261, 351, 352, 436, 510, 518, 521, 691, 796]


def load_tasks(path):
    """Return {task_id: intent}. Accepts a top-level list or a dict wrapping one."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for k in ("tasks", "data", "items"):
            if isinstance(data.get(k), list):
                data = data[k]
                break
        else:
            raise SystemExit(f"Could not find a task list inside {path} (top-level keys: {list(data)})")
    tasks = {}
    for t in data:
        if "task_id" not in t:
            raise SystemExit(f"A task entry is missing 'task_id': {str(t)[:120]}")
        tasks[int(t["task_id"])] = t.get("intent", "")
    if not tasks:
        raise SystemExit(f"No tasks loaded from {path}")
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="all_shopping_tasks.json", help="path to the task list JSON")
    ap.add_argument("--out-dir", default=".", help="where to write splits.json + checklist")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.70, help="train fraction of all tasks")
    ap.add_argument("--val", type=float, default=0.15, help="val fraction of all tasks (test = remainder)")
    ap.add_argument("--demos", type=int, default=3, help="human demos to collect per train/val task")
    args = ap.parse_args()

    if args.train + args.val >= 1.0:
        raise SystemExit("--train + --val must be < 1.0 (test is the remainder)")

    tasks = load_tasks(args.tasks)
    all_ids = sorted(tasks)                       # deterministic regardless of file order
    n = len(all_ids)

    # Every forced eval id must actually exist in the dataset.
    missing = [i for i in EVAL_TASK_IDS if i not in tasks]
    if missing:
        raise SystemExit(f"These forced eval IDs are NOT present in {args.tasks}: {missing}")

    eval_set = set(EVAL_TASK_IDS)
    pool = [i for i in all_ids if i not in eval_set]     # free to assign

    # Target sizes over the WHOLE dataset.
    n_train = round(args.train * n)
    n_val = round(args.val * n)
    n_test = n - n_train - n_val

    if n_test < len(eval_set):
        print(f"NOTE: target test size ({n_test}) < forced eval IDs ({len(eval_set)}); "
              f"test will be enlarged to {len(eval_set)} to fit them.")

    # test already holds the forced eval IDs; fill the rest from the shuffled pool.
    n_test_extra = max(0, n_test - len(eval_set))

    rng = random.Random(args.seed)
    shuffled = pool[:]
    rng.shuffle(shuffled)

    test_extra = shuffled[:n_test_extra]
    rest = shuffled[n_test_extra:]
    train = rest[:n_train]
    val = rest[n_train:n_train + n_val]
    leftover = rest[n_train + n_val:]                    # rounding remainder (usually empty) -> test

    train = sorted(train)
    val = sorted(val)
    test = sorted(eval_set | set(test_extra) | set(leftover))

    # ---- invariants: fail loudly rather than ship a bad split ----
    assert len(train) + len(val) + len(test) == n, "splits do not sum to N"
    assert not (set(train) & set(val)), "train/val overlap"
    assert not (set(train) & set(test)), "train/test overlap"
    assert not (set(val) & set(test)), "val/test overlap"
    assert eval_set <= set(test), "a forced eval ID escaped test"
    assert not (eval_set & set(train)), "a forced eval ID leaked into train"
    assert not (eval_set & set(val)), "a forced eval ID leaked into val"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "seed": args.seed,
        "n_tasks": n,
        "fractions": {"train": args.train, "val": args.val,
                      "test": round(1 - args.train - args.val, 4)},
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "demos_per_task": args.demos,
        "forced_eval_ids_in_test": sorted(eval_set),
        "train": train,
        "val": val,
        "test": test,
    }
    (out / "splits.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with open(out / "collection_checklist.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "split", "demos_needed", "intent"])
        for split_name, ids in (("train", train), ("val", val)):
            for i in ids:
                w.writerow([i, split_name, args.demos, tasks[i]])

    demo_tasks = len(train) + len(val)
    print(f"Loaded {n} tasks from {args.tasks}  (seed={args.seed})")
    print(f"  train: {len(train):>3}  ({100*len(train)/n:4.1f}%)")
    print(f"  val  : {len(val):>3}  ({100*len(val)/n:4.1f}%)")
    print(f"  test : {len(test):>3}  ({100*len(test)/n:4.1f}%)   incl. {len(eval_set)} forced eval IDs")
    print(f"\nDemos to collect: {demo_tasks} tasks x {args.demos} = {demo_tasks*args.demos} "
          f"trajectories (train+val only).")
    print(f"Test = automated WebArena eval, no demos.")
    print(f"\nWrote:\n  {out/'splits.json'}\n  {out/'collection_checklist.csv'}")


if __name__ == "__main__":
    main()
```

---

## 9. Collection workflow

**Login setup (do once; repeat when cookies expire).** The recorder loads `./shopping_state.json`. Cookies expire and are host-scoped, so the file copied from the Alienware can fail (it failed during this project — either expired or wrong host). Regenerate with `make_login_state.py`, which logs in at the **exact recorder URL** so the cookie domain matches. Run it from `data_collection/`, watch for **"Sign Out"** in the browser top-right, then press Enter to save. If a session opens to "Sign In", just regenerate and continue. (Verified working: after regeneration, the recorder showed "Sign Out" and `Loading logged-in state from ./shopping_state.json`.)

### `make_login_state.py` (full source)
```python
#!/usr/bin/env python3
"""
Regenerate the logged-in storage state the recorder loads (./shopping_state.json).

Why this exists: cookies in the storage state expire, and they're scoped to a
specific host. This logs in at the SAME url the recorder uses, so the saved
cookies match the domain the recorder will navigate to. Run it from
data_collection/ (so ./shopping_state.json lands next to record.py).

It auto-fills the Magento login form (best effort). If the selectors ever miss,
the browser stays open -- just sign in by hand, then press Enter.

  python make_login_state.py
"""
from playwright.sync_api import sync_playwright

# Must match the recorder's TASK_URL exactly (same host => cookies apply).
BASE = "http://liralabwidowx-alienware-aurora-r16.tail4d611e.ts.net:8082"
EMAIL = "emma.lopez@gmail.com"
PASSWORD = "Password.123"
OUT = "./shopping_state.json"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    ctx = browser.new_context()
    page = ctx.new_page()
    page.goto(f"{BASE}/customer/account/login/", wait_until="domcontentloaded")

    # Best-effort auto-login (standard Magento Luma selectors).
    try:
        page.fill("#email", EMAIL, timeout=5000)
        page.fill("#pass", PASSWORD, timeout=5000)
        page.click("#send2", timeout=5000)
        page.wait_for_load_state("networkidle", timeout=20000)
    except Exception as e:
        print(f"(auto-fill didn't complete: {e})")
        print("Sign in by hand in the browser window.")

    print("\n>>> Check the browser: top-right should show 'Sign Out' (= you're logged in).")
    print(">>> If it still shows 'Sign In', log in manually now (emma.lopez@gmail.com / Password.123).")
    input(">>> Then press Enter here to save the login state... ")

    ctx.storage_state(path=OUT)
    print(f"Saved login state to {OUT}")
    browser.close()
```

**The per-task loop.** Work one task at a time, all 3 demos back-to-back:
1. `python progress.py` → it names the next task id + intent.
2. Read the intent so you know the goal and the answer string.
3. `python record.py` → enter the **bare integer task id** and trajectory number → perform the task → close the browser → type the answer at the prompt.
4. Repeat trajectory 2, 3; then move to the next task.

Do the 3 demos as genuine attempts (natural variation in path/scroll is good); each must actually complete the task. Fumbled a demo? Answer `n` to "Keep this recording?" and redo.

**Per-session habits (a sitting of ~15–25 tasks):**
- **Calibrate** GazePoint at the start; keep eyes on the **browser** during clicks/scrolls; don't move/resize the window mid-recording.
- **Re-check login** at session start (cookies expire over days).
- **Back up** `recordings/` to the Alienware at session end — it's the only copy of real labor and where preprocessing reads from. `scp -r recordings liralab-widowx@100.106.3.51:/data/gaze-web-nav-training/recordings`; switch to rsync (Git Bash) for incremental once it's large.

### `progress.py` (full source)
```python
#!/usr/bin/env python3
"""
Collection progress tracker for the gaze-web-nav demos.

Scans recordings/ against collection_checklist.csv and reports which train/val
tasks are complete (all --demos trajectories present), in progress, or not
started -- plus the next tasks to record.

A trajectory folder task{id}_traj{n} COUNTS only if it has a non-empty
actions.json AND a non-empty gaze.json, so empty/aborted runs and any gaze-off
runs are never miscounted as done.

Usage:
  python progress.py
  python progress.py --next 12
  python progress.py --recordings ./recordings --checklist collection_checklist.csv
"""
import argparse
import csv
import json
from pathlib import Path


def traj_ok(folder, require_gaze=True):
    """A trajectory counts if actions.json is a non-empty list (and gaze too, unless disabled)."""
    a = folder / "actions.json"
    if not a.exists():
        return False
    try:
        acts = json.loads(a.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not (isinstance(acts, list) and len(acts) > 0):
        return False
    if require_gaze:
        g = folder / "gaze.json"
        if not g.exists():
            return False
        try:
            gaze = json.loads(g.read_text(encoding="utf-8"))
        except Exception:
            return False
        if not (isinstance(gaze, list) and len(gaze) > 0):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", default="./recordings")
    ap.add_argument("--checklist", default="collection_checklist.csv")
    ap.add_argument("--demos", type=int, default=3)
    ap.add_argument("--next", type=int, default=8)
    ap.add_argument("--no-gaze-check", action="store_true",
                    help="count a trajectory even if gaze.json is empty")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.checklist, encoding="utf-8")))
    by_id = {int(r["task_id"]): r for r in rows}
    rec = Path(args.recordings)
    require_gaze = not args.no_gaze_check

    done, partial, todo, total_traj = [], [], [], 0
    for tid, r in by_id.items():
        have = sum(1 for n in range(1, args.demos + 1)
                   if (rec / f"task{tid}_traj{n}").is_dir()
                   and traj_ok(rec / f"task{tid}_traj{n}", require_gaze))
        total_traj += have
        if have >= args.demos:
            done.append(tid)
        elif have > 0:
            partial.append((tid, have))
        else:
            todo.append(tid)

    n_tasks = len(rows)
    target = n_tasks * args.demos
    print(f"Tasks : {len(done)}/{n_tasks} complete  "
          f"({len(partial)} in progress, {len(todo)} not started)")
    bar = int(40 * total_traj / target) if target else 0
    print(f"Demos : {total_traj}/{target}  [{'#'*bar}{'.'*(40-bar)}]  "
          f"{(100*total_traj/target) if target else 0:.1f}%")

    if partial:
        print("\nFinish these (already started):")
        for tid, have in sorted(partial):
            print(f"  task {tid:<5} {have}/{args.demos}  [{by_id[tid]['split']}]  "
                  f"{by_id[tid]['intent'][:62]}")

    queue = [tid for tid, _ in sorted(partial)] + sorted(todo)
    if queue:
        print(f"\nNext {min(args.next, len(queue))} to record:")
        for tid in queue[:args.next]:
            print(f"  task {tid:<5} [{by_id[tid]['split']}]  {by_id[tid]['intent'][:62]}")
    else:
        print("\n*** Collection complete - every train/val task has all demos. ***")

    # flag stray folders not in the checklist (e.g. leftover task9999 test runs)
    if rec.is_dir():
        valid = {f"task{t}_traj{n}" for t in by_id for n in range(1, args.demos + 1)}
        strays = sorted(p.name for p in rec.iterdir()
                        if p.is_dir() and p.name.startswith("task") and p.name not in valid)
        if strays:
            shown = ", ".join(strays[:8]) + (" ..." if len(strays) > 8 else "")
            print(f"\nNote: {len(strays)} folder(s) not in the checklist "
                  f"(test runs / extra demos): {shown}")


if __name__ == "__main__":
    main()
```

---

## 10. Methodology decisions to preserve

**Perform-while-gazing — do NOT automate the actions.** The action label already encodes *where* the action happened; gaze is valuable only because it captures *where attention went while deciding* (the products scanned, prices compared, regions ruled out). That signal exists only when a human is actually doing the deciding. If a script acted and the human just watched, gaze would flip from *leading* the action to *following* the cursor, the ~200 ms-before-decision window would be meaningless, and on the highest-value tasks ("find cheapest X", "which matches") the comparison behavior would never be performed. It's also not actually faster: automating actions needs the correct action sequence per task — which is exactly the BC target you're collecting (no oracle exists). Misclicks are cheap (the keep-y/n redo). The standard gaze-imitation setup records gaze + actions simultaneously from the same performing human — which is the setup already in place. (Worth re-confirming with Yutai, but the answer is settled for this project.)

**Gaze masking.** Apply the gaze KL **only** to spatially-grounded actions: `click`, `left_double`, `right_single`, `drag`, `scroll`. **Skip** `type`, `hotkey`, `finished` (during typing the eyes are on the keyboard → invalid/off-screen gaze; these actions have no screen target). `FPOGV` + in-bounds filtering removes invalid samples everywhere as a second line of defense.

**Gaze windowing.** Aggregate gaze in a window of roughly **200 ms before** each action's decision-moment `timestamp`. The exact window size is **to be confirmed with Yutai**. (For `type`, the timestamp anchors to the flush, not focus — minor, and type gaze is skipped anyway.)

**Coordinates.** Absolute pixels via ByteDance `smart_resize`, never 0–1000 normalization.

---

## 11. Pending work (future phases — none of these require re-collecting data)

**A. `preprocess.py` overhaul** (the OLD version is wrong/incomplete):
- Map the new action types: `left_double`, `right_single`, `drag`, `hotkey`, scroll-with-coords, `finished` (and native `select` if used).
- Coordinates **0–1000 → absolute pixels** (`smart_resize`).
- Confirm the hotkey string format the model expects: `"ctrl a"` vs `"ctrl+a"` vs the OSWorld parser convention.
- Decide native `<select>` mapping.
- Drop `start`/`navigate` as action labels (use them only for trajectory segmentation).
- Join task intent from `all_shopping_tasks.json` by integer task id.
- Build the gaze maps: filter `FPOGV==1` + in-bounds, window ~200 ms pre-decision, apply only to spatially-grounded actions, map gaze → screenshot pixels per §6.

**B. Training:**
- Add `eval_steps` to the training config (currently missing).
- Model **B**: behavioral cloning LoRA on screenshots + actions (the old `outputs/bc_baseline/` is obsolete; retrain on the new dataset/format).
- Model **C**: custom training loop adding the gaze KL — last layer only, mode `mean_then_kl`, λ ≈ 0.001.

**C. Evaluation** (fixes needed in the OLD scripts; current main blockers):
- `run_eval.py`: the `evaluator_router` is passed a config **dict** but needs the config **file path** (beartype error zeros scores).
- Fix `observation_type` → `"accessibility_tree"`; `obs["screenshot"]` → `PIL.Image.fromarray(obs["image"])`.
- `uitars_agent.py`: `merge_and_unload()` crashes (accelerate 1.11.0, "unhashable type: set") — instead load 4-bit + `PeftModel.from_pretrained` **without** merging.
- Then eval A and B; implement and eval C; compare (target C > B > A).
- Remember the transformers pin conflict (§2 caveat) for the eval env.

**Low-risk, untested recorder paths** (flagged, not blockers): logged-in start via `storage_state` (now verified working), element-level scroll (`is_page=False`), multi-tab/popups.

---

## 12. Immediate next step

1. Confirm logged in (open recorder once, see "Sign Out"; regenerate with `make_login_state.py` if not).
2. `python progress.py` → get the first task.
3. Record trajectory 1/2/3 for it (bare integer id), then the next task. Calibrate at session start; back up `recordings/` to the Alienware at session end.

Everything the researcher was anxious about redoing — recorder, data format, split, login flow, progress tracking — is locked. The next real work is the collection itself, then the preprocess overhaul (§11A).
