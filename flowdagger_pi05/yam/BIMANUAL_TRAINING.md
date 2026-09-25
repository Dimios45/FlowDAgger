# Bimanual YAM training runbook

## Current configuration: 34 autonomous + 20 HITL

Use [yam_cloth_all_34auto_20hitl_50.json](configs/yam_cloth_all_34auto_20hitl_50.json):

```bash
cd /home/sra/tirth/FlowDAgger
bash flowdagger_pi05/yam/run_bimanual.sh --config flowdagger_pi05/yam/configs/yam_cloth_all_34auto_20hitl_50.json
```

| Setting | Value |
|---|---|
| Epochs / checkpoint | 50 full epochs; `actor.pt` ends at zero-based epoch 49 |
| Holdouts / early stopping | None; validation and test evaluation disabled |
| Optimizer | AdamW, constant learning rate 3e-5, weight decay 1e-3 |
| Batch / objective | 32 human + 32 autonomous windows, equal-weight noise MSE |
| Gradient clipping / seed | 1.0 / 0 |
| Actor initialization | Fresh actor; not a continuation of the previous actor |
| Base / features | Frozen original pi0.5 and visual features |
| Autonomous timing | 30 consecutive recorded commands; window stride 30; no resampling |
| Human timing | 30 consecutive human commands within one intervention; stride 10 |

A batch contains **windows, not episodes**. Each epoch visits the accepted human
buffer once; autonomous windows are sampled with replacement in equal numbers.
The smaller final batch maintains equal shares. Hyperparameters are engineering
choices for this task, not paper-specified YAM settings or a proven optimum.

## Data and provenance

| Source | Pinned Hugging Face revision | Use |
|---|---|---|
| [50 autonomous episodes](https://huggingface.co/datasets/Dimios45/pi05-molmoact2-yam-autonmous-buffer) | `4745f920c6903dd6afe85a52896384f7a436ba4b` | 34 eligible successful episodes |
| [20 HITL episodes](https://huggingface.co/datasets/Dimios45/pi05-pink-towel-dagger6) | `7804a2a78ddc2c733a5bc01840a36b5cbdb2bd45` | Human-control windows from all 20 episodes |

File sizes, Git blob hashes and LFS SHA256 hashes are committed in `snapshots/`.
The autonomous upload has 35 success labels: 15 failures are excluded, and one
successful episode with a different instruction (`fold the towel` plus a trailing
backslash) is excluded. Outcome labels are recorded operator judgments, not an
independent visual certification. Human corrections remain eligible regardless
of the containing episode's final outcome. Never cross episode/intervention
boundaries, fill incomplete windows or substitute future states for commands.

The original human cache retains old 16/2/2 split labels as historical provenance.
`all_training` explicitly includes accepted windows from all three partitions;
they are no longer holdouts. The autonomous all-training manifest has train=34,
validation=0, test=0. The reviewed episode-12 reset range [4900,5025) remains
excluded from human training. The three later hardware HITL episodes are not in
the pinned 20-episode upload and are not included in this run.

## Recorded timing

No 0.5-to-1 temporal conversion is applied in the current configuration. Targets
retain consecutive commands as recorded at nominal 30 Hz. Autonomous metadata
reports playback speed 0.5. HITL metadata reports 1.0, despite the user's recollection
of 0.5; inspected Karma applies playback interpolation to policy chunks, not
human teleoperation commands.

Karma client `--speed 0.5` independently stretches 30 predicted actions into 59
executed ticks. It will still do so when the actor was trained on already slowed
recorded commands. The current experiment intentionally preserves recorded timing;
it is not temporally equivalent to reconstructing native-speed model chunks.
PTS/gap checks establish encoded frame alignment, not wall-clock synchronization
or the original inference request observation.

Dataset gripper polarity remains open=0. Model/wire polarity remains open=1.
The original audit and autonomous episode-0 frames 0 (open) and 1500 (closed)
support this mapping. Numeric channels 6 and 13 are flipped for both state and
actions before saved normalization; joint targets remain absolute.

## Local dependencies and reproduction

The wrapper uses these existing workstation assets:

- `/home/sra/molmoact2/.venv-pi05-server/bin/python` and native LeRobot source.
- Base: `/home/sra/molmoact2/outputs/models/molmoact2-yam-pi05`.
- Tokenizer: `/home/sra/molmoact2/outputs/models/paligemma-tokenizer`.
- `outputs/yam-flowdagger-v1/manifest.json` and complete `cache/`.
- `outputs/yam-flowdagger-v1/preservation-review/review.json` with source-manifest
  identity, reset exclusion and local evidence.
- Verified gripper conventions embedded in the human manifest; serving uses
  `outputs/yam-gripper-audit/conventions.json`.

These local artifacts are ignored by Git. Preserve them separately to reproduce
this exact run. A fresh clone needs a gripper audit, human preparation/inversion
and a review tied to that manifest before using this wrapper. The individual
CLI exposes `audit`, `prepare`, `invert` and `verify`; use `--help` for arguments.
The wrapper verifies human source files against the pinned upload and downloads
or reuses the autonomous snapshot. Moving datasets or changing runtime identities
may require rebuilding manifests/caches; never bypass identity checks.

## Execution, restart and outputs

The wrapper verifies sources, prepares windows, loads the frozen model, inverts
missing targets, runs numerical verification, and trains. It does not deploy or
command hardware. A lock prevents concurrent wrapper runs.

- Human inversion cache is reused unchanged.
- Current autonomous cache: `outputs/yam-flowdagger-v2/recorded-cadence/all-training/autonomous-cache/`.
- `latest-run.txt` under `outputs/yam-flowdagger-v2/` identifies the newest run.
- Each run contains `resolved-config.json`, `preparation.json`, `verification.json`,
  `buffer-counts.json`, `actor.training.json`, `actor.history.json`, `run.log`,
  `status.json`, and `actor.pt`.
- Epoch logs show **training noise MSE**, not validation error. There is no
  `comparison.json` in all-training mode.
- `actor.pt` is updated atomically after each epoch. At successful completion it
  contains epoch 49. An interrupted run may contain an earlier epoch.
- Restarting the command creates a new run, reuses compatible completed inversion
  targets, and trains a fresh actor. Optimizer/epoch training resume is not implemented.

`Verified ...` messages indicate local hash checks, not repeated downloads.
Changing target frame indices requires new inversion labels; changing the number
of epochs does not. Caches for the older timing modes remain separate.

## Completed final run, September 25, 2026

Run: `outputs/yam-flowdagger-v2/training/bimanual-20260925T123943Z-lTrWYA/`.

| Artifact measurement | Value |
|---|---|
| Accepted human training windows | 1,746 across 20 episodes |
| Autonomous candidate / accepted windows | 9,580 / 9,325 across 34 episodes |
| Completed epochs | 50; saved epoch 49 |
| Final training noise MSE | 0.15898472500830582 |
| Validation/test evaluation | Neither performed |
| Robot commands during training | None |

The model is a completed training artifact, not a validated physical-success
claim. All eligible recorded episodes were used for training; use new rollouts
for independent evaluation. [Deployment commands](DEPLOYMENT.md) select this actor.

## Earlier experiments (not the current configuration)

| Mode | Invocation | Difference |
|---|---|---|
| Native-spacing reconstruction | `run_bimanual.sh` without a config | Infers a 59-frame playback phase, uses every second frame, filters interpolation residuals, keeps holdouts, best validation checkpoint |
| Recorded cadence with holdouts | `--config flowdagger_pi05/yam/configs/yam_cloth_recorded_50.json` | Consecutive targets, 50 epochs, last checkpoint, validation still evaluated |
| Current all-training | `--config flowdagger_pi05/yam/configs/yam_cloth_all_34auto_20hitl_50.json` | Consecutive targets, 34+20 episodes, no holdouts, last checkpoint |

The earlier native-spacing run `bimanual-20260925T101347Z-ybUoUi` selected epoch 3.
Its equal-buffer validation RMSE was 0.15920 (zero-noise 0.16118, random-noise
0.20295). The user reported better hardware behavior with that actor; no controlled
success-rate result is available. These values cannot be compared directly with
the final run's training noise MSE or its changed target timing.
