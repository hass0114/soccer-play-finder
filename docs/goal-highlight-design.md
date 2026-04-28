# Goal Highlight Generator with Scorer Attribution — Design

End-to-end system that takes a single broadcast soccer match video (MP4) as input and produces a structured game log of every goal, including the scoring player's name and start/end timestamps of a clip that begins with the buildup play and ends after the celebration.

---

## 1. Target output

```json
{
  "goals": [
    {
      "goal_id": 1,
      "scorer": "Bukayo Saka",
      "scorer_jersey": 7,
      "team": "Arsenal",
      "clip_start": 1245.3,   // play buildup begins
      "goal_time": 1268.7,    // ball enters net
      "clip_end":  1305.2,    // celebration ends
      "confidence": 0.91,
      "evidence": ["score_change", "crowd_roar", "replay_cluster"]
    }
  ]
}
```

---

## 2. Pipeline overview

```
MP4 ──► [A] Detection+Tracking ──► [B] Goal event detector ──┐
   │                                                          │
   ├──► [C] Audio analysis ──────────────────────────────────►│
   │                                                          ├──► [F] Fusion ──► [G] Clip
   ├──► [D] Scoreboard OCR ──────────────────────────────────►│         │           boundaries
   │                                                          │         │
   └──► [E] Scorer attribution ◄──── (around each goal time) ─┘         │
                                                                        ▼
                                                                 Game log JSON
```

---

## 3. Models per stage

| Stage | Task | Model | Why |
|---|---|---|---|
| A1 | Player/ball/ref detection | **YOLOv8m** fine-tuned on SoccerNet | Domain-specific classes (`player`, `goalkeeper`, `referee`, `ball`, `goalpost`) |
| A2 | Multi-object tracking | **ByteTrack** or **BoT-SORT** (built into ultralytics) | Stable IDs across frames; required for scorer attribution |
| A3 | Pitch keypoints / homography | **PnLCalib** or **TVCalib** (open-source) | Maps pixel coords → field coords; lets you say "ball entered goal mouth" |
| B  | Ball-in-goal geometric check | Trajectory + goalpost bbox | Direct evidence of goal |
| C1 | Crowd roar / commentator excitement | **PANNs (CNN14)** or **YAMNet** | Pretrained audio event classifiers; YAMNet has a "cheering"/"crowd" class out of the box |
| C2 | Whistle | Bandpass + peak detection (already in pipeline) | Cheap; useful to detect kickoff after goal |
| D1 | Scoreboard region localization | Temporal pixel variance + small UNet OR fine-tuned YOLO class | Scoreboards are static overlays |
| D2 | Score OCR | **PaddleOCR** (PP-OCRv4) or **TrOCR** | Best for small overlay text |
| E1 | Jersey number recognition | **PARSeq** scene-text recognizer fine-tuned on jersey crops | Reads the back of the shirt |
| E2 | Team classification | K-means in HSV on torso crops + manual roster mapping | Cheap and effective |
| E3 | Action spotting (who shot the ball) | **T-DEED** or **E2E-Spot** action spotting model | State-of-the-art for "shot" event localization on SoccerNet |
| E4 (optional) | Player face/jersey ReID | **OSNet** or **CLIP-ReID** fine-tuned on player crops | Re-identify scorer in celebration close-ups |
| F  | Multi-signal fusion | Hand-tuned scoring + small **XGBoost** classifier | Combine audio/visual/OCR signals |

You can start with **A + C + D** only — that already gets reliable goal detection. Add **E** for scorer attribution.

---

## 4. Training data

### A. Player/ball detection (YOLOv8 fine-tune)
- **SoccerNet-v3** (free, academic): ~500k labeled frames with player/ball/ref bounding boxes across 500+ matches.
- **Roboflow "football-players-detection-3zvbc"**: smaller, ~2k images, easy to start with, includes goalkeeper class.
- **SoccerTrack** dataset: drone + broadcast footage with tracking IDs.

If you only need broadcast footage, ~5–10k frames hand-labeled (or auto-labeled with SAM2 + manual cleanup) gets you a usable model.

### B. Goalpost detection
- Add a `goalpost` class to your YOLO labels. Hand-label ~500 frames showing the goal frame at various angles.
- Or use **SoccerNet calibration** dataset which includes goal landmarks.

### C. Audio (crowd roar / commentator)
- **AudioSet** (Google): pretrained YAMNet/PANNs already cover "cheering", "crowd", "applause", "shout".
- For sport-specific tuning, extract 1000 short clips (5s) from any SoccerNet match: positive = around known goal timestamps (provided in SoccerNet labels), negative = random match audio.
- Train a small MLP head on YAMNet embeddings (~30 min on a laptop CPU).

### D. Scoreboard OCR
- **PaddleOCR is zero-shot** on most TV scoreboards — usually no training needed.
- If it struggles on a specific broadcast style: collect 200 cropped scoreboard images from that broadcaster, fine-tune **TrOCR** (HuggingFace `microsoft/trocr-small-printed`).

### E. Jersey number recognition
- **SoccerNet Jersey Number Recognition challenge** dataset: 2.8k tracklets with ground-truth jersey numbers — purpose-built for this.
- Fine-tune **PARSeq** on jersey crops (resize to 64×128, augment with rotation/blur to simulate motion).
- Expect ~85% accuracy per frame; aggregate across the tracklet (mode of 30+ predictions per player) → >95%.

### F. Action spotting (locating "shot"/"goal" event precisely)
- **SoccerNet-v2 Action Spotting**: 17 event classes including `Goal`, `Shots on target`, `Shots off target`, `Penalty`. 500 matches, dense annotations.
- Train **T-DEED** (CVPR 2024, SOTA on SoccerNet) — gives you sub-second event localization.

### G. Roster mapping (jersey → name)
- Need an external lookup: scrape the team sheet for the match (Wikipedia, the league's API, or pass it in as a CLI arg `--roster roster.json`).
- Without rosters you can only output `"#7 Arsenal"`, not `"Bukayo Saka"`.

---

## 5. Scorer attribution algorithm

For each detected goal at time `t_goal`:

1. **Look back 1–5 seconds before `t_goal`**: find the player track whose bbox is closest to the ball at the moment the ball's velocity spikes toward the goal (the "shot" frame).
2. Confirm via **action spotter** — T-DEED outputs a "Shot" event timestamp; pick the player nearest the ball at that exact frame.
3. **Read jersey number** from that track's bbox crops: take 30 frames around the shot, run PARSeq on the back-of-jersey region (estimated as upper-half of bbox), take the **modal prediction** across frames where confidence > 0.7.
4. Determine **team** via torso-color cluster.
5. **Look up roster**: `(team, jersey_number) → player_name`.
6. (Optional fallback) If jersey unreadable: in the celebration sequence (5–20s after goal), find the player with highest "celebration pose" score (arms raised, running away from goal); jersey is often more readable in close-ups.

---

## 6. Clip boundary detection

For each goal at `t_goal`:

**Clip start (`t_goal - 30s` refined):**
- Walk backwards from `t_goal` and find the last `PlayState` transition into `ACTIVE_PLAY`.
- Or simpler: `t_goal - max(15, time_since_last_stoppage)`, capped at 45s.

**Clip end (`t_goal + 30s` refined):**
- Walk forward from `t_goal` until **all** are true:
  - Crowd audio energy returns to baseline,
  - No replays detected for 5+ seconds,
  - `PlayState` returns to `ACTIVE_PLAY` or kickoff detected (players in own halves, ball on center spot).
- Cap at 90s post-goal.

---

## 7. Concrete code modules

| File | Class | Purpose |
|---|---|---|
| `detector.py` | `SoccerDetector` | Replace `torch.hub` YOLOv5 with `ultralytics.YOLO("soccernet_yolov8.pt").track(...)` |
| `pitch.py` | `PitchCalibrator` | Homography via PnLCalib; provides `pixel_to_field()` |
| `goal_detector.py` | `GoalDetector` | Geometric ball-in-goal check |
| `audio.py` | `CrowdAudioAnalyzer` | YAMNet-based crowd/commentator detection (extends `AudioAnalyzer`) |
| `scoreboard.py` | `ScoreboardOCR` | Detect overlay region + PaddleOCR loop |
| `attribution.py` | `ScorerAttributor` | Jersey OCR (PARSeq) + team color + roster lookup |
| `action_spotter.py` | `ActionSpotter` | T-DEED inference for precise shot timing |
| `fusion.py` | `GoalEventFuser` | Combine all signals into final `GoalEvent` list |
| `clip_builder.py` | `ClipBuilder` | Refine start/end times around each goal |

Update `Config` to add: roster path, jersey-OCR confidence, audio model paths, etc.

`main()` becomes:

```python
detector.run(video) → tracks, ball_traj, goalpost_dets
scoreboard.run(video) → score_change_events
audio.run(video) → crowd_events, whistle_events
action_spotter.run(video) → shot_events
goal_events = fusion.fuse(...)
for ev in goal_events:
    ev.scorer = attribution.identify(ev, tracks, roster)
    ev.clip_start, ev.clip_end = clip_builder.bounds(ev, state_machine)
write_json(goal_events)
```

---

## 8. Recommended build order

Each step yields a working system on its own.

1. **Switch to YOLOv8 + ByteTrack** (1 day). Stable player IDs.
2. **Scoreboard OCR** (1–2 days). Single most reliable goal signal.
3. **Audio crowd detection with YAMNet** (1 day). Confirms #2.
4. **Clip boundary refinement** using existing `PlayStateMachine`. (½ day)
5. **Jersey number recognition with PARSeq** + roster JSON (3–5 days). Now you have scorer names.
6. **Action spotter (T-DEED)** for precise goal timestamp. (2–3 days)
7. **Geometric ball-in-goal verification** + goalpost detector (3–5 days).
8. **Fusion classifier** trained on a few labeled matches (½ day).

---

## 9. Datasets summary

**SoccerNet** (https://www.soccer-net.org) is the one-stop shop:

- 500+ broadcast matches with goal/event timestamps,
- Player/ball detection labels,
- Jersey number tracklets,
- Camera calibration ground truth,
- Action spotting labels.

Sign up for an academic license (free), download with `pip install SoccerNet` + their CLI. Low-res (224p) videos and features download immediately; high-res (720p) videos require a free password obtained via their NDA form.

For roster/player names: scrape per-match from FBref, Sofascore API, or pass in manually via `--roster roster.json`.

---

## 10. Evaluation

**Goal detection** — SoccerNet-style precision/recall and average-mAP at $\delta = 5\,\text{s}$ tolerance (a predicted timestamp counts as a TP if within ±5 s of ground truth).

**Scorer attribution** — top-1 player-name accuracy, conditional on the goal being correctly detected.

**Clip boundaries** — mean absolute error of `(start, end)` in seconds plus temporal IoU against human-annotated clips.

**Ablations** — disable each modality (OCR / audio / action spotter / detector-only) one at a time to quantify its contribution.
