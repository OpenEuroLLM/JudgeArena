---
title: JudgeArena Leaderboard
emoji: 🏆
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# JudgeArena Leaderboard

Assembles the leaderboard at load time from the JudgeArena results dataset — the
per-model records under `records/{panel_version}/` plus the precomputed panel
caches under `panel/{panel_version}/` — and renders it. No prebuilt
`leaderboard.json`; no Bradley-Terry or inference code runs in the Space.

Local preview:

    python space/app.py --local <dataset-dir>    # dir containing panel/ and records/
    python space/app.py --repo <user/dataset>    # pull from the HF dataset

**Deployment note:** the Space is self-contained from three files — `app.py`,
`render.py`, and `assemble.py` (copied next to `app.py`). `assemble.py` imports
no `judgearena`/Bradley-Terry code, so the Space needs only the packages in
`requirements.txt`. The `deploy-space.yml` workflow syncs these on push to main.
