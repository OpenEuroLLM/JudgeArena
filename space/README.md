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

Renders the published leaderboard bundle (`leaderboard.json` + `scores.parquet`)
from the JudgeArena results dataset.

Local preview:

    python space/app.py --local <dataset-dir>    # dir containing panel/ and records/

**Deployment note:** the deployed Space must have `assemble.py` (and its `anchors` dependency) available next to `app.py` — either copy `judgearena/leaderboard/assemble.py` and `judgearena/leaderboard/anchors.py` into the Space, or ship the `judgearena` package as a Space dependency.
