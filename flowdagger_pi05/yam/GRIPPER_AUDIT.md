# Gripper convention audit — 2026-09-21

The inspected dataset uses **0=open, 1=closed**. The planned original
`/home/sra/molmoact2/outputs/models/molmoact2-yam-pi05` checkpoint uses
**1=open, 0=closed**, as does the inspected Karma inference path.
Convert dataset channels **6 and 13 with `1 - value` for both state and actions**
before checkpoint preprocessing. Original-model output to this Karma wire is an
identity conversion. Joint channels are untouched.

## Local correction dataset: visual and code evidence

Dataset: `outputs/yam-flowdagger/dataset`. The following episode-0 frames were
extracted using recorded video offsets/PTS and inspected in all three camera views.
Numbers below are `[left, right]` grippers.

| Frame | Measured state | Commanded action | Visual interpretation |
|---|---|---|---|
| 30 | [0.000596, 0.000513] | [0.000000, 0.000525] | Both open |
| 4219 | [0.977364, 0.970083] | [1.000000, 1.000000] | Both closed, holding cloth |
| 3563 | [0.000670, 0.985042] | [0.002139, 0.999497] | Left open, right gripping cloth |
| 2322 | [0.979672, 0.008432] | [0.993556, 0.003392] | Left gripping cloth, right open |

Contact sheets: [both open](../../outputs/yam-gripper-audit/low-low.png),
[both closed](../../outputs/yam-gripper-audit/high-high.png),
[left open/right closed](../../outputs/yam-gripper-audit/low-high.png),
[left closed/right open](../../outputs/yam-gripper-audit/high-low.png).

Karma's `src/openpi_control/record.py:93` explicitly defines native open as 1 and
dataset open as 0. `to_dataset_gripper`/`to_native_gripper` implement `1 - x`.
`observation_vector` at line 343 converts measured state; `action_vector` at
line 378 converts commanded targets. Uncommanded grippers retain the already
converted observation value.

## Original checkpoint: independently checked source-training frames

The saved `train_config.json` identifies
`allenai/MolmoAct2-BimanualYAM-Dataset` as its training dataset. Its saved processors
contain quantile normalization/unnormalization and no gripper inversion. The model
card's `[0,1]` range alone was not used to infer polarity.

Downloaded the first public training data shard and episode metadata at source
revision `e9f21ae15074330839f2ac25ed4b49d76dfa1f9c`, and decoded original top-camera
frames over HTTP:

| Episode 0 frame | State [left,right] | Action [left,right] | Visual evidence |
|---|---|---|---|
| 0 | [0.998628, 0.999349] | [0.951789, 0.991234] | Both jaws open |
| 984 | [0.953361, 0.010341] | [0.953981, 0.013869] | Left open, right closed |
| 1496 | [0.054653, 0.991973] | [0.055506, 0.993426] | Left closed around block, right open |

Images: [open](../../outputs/yam-gripper-audit/source-training/frame-0.png),
[right closed](../../outputs/yam-gripper-audit/source-training/frame-984.png),
[left closed](../../outputs/yam-gripper-audit/source-training/left-closed.png).
[Versioned training dataset](https://huggingface.co/datasets/allenai/MolmoAct2-BimanualYAM-Dataset/tree/e9f21ae15074330839f2ac25ed4b49d76dfa1f9c).
The checkpoint records training dataset revision as `local`, so this verifies the
published source convention; it is not a byte-for-byte audit of its original
training machine's dataset snapshot.

## Karma wire and software tests

Inspected checkout: `/home/sra/ksagar/karma`, HEAD
`eae105f233b6d53fbece42f8b902ba7a67719f34`. The checkout contains unrelated local
changes; `record.py`, `inference.py`, and `types.py` have no local diff from HEAD.

`inference.py:601` packs native measured grippers directly into the HTTP state.
The payload at line 425 carries this state without a flip. `_commands_from_row`
at line 745 and `BoundedChunkExecutor.step` preserve response polarity; the
latter rate-limits/clips values without inverting them. Line 139 documents native
open=1. Both camera and left/right action orders match the new backend.

Ran [check_karma_conversions.py](../../outputs/yam-gripper-audit/check_karma_conversions.py):
25 paired combinations of {0,.25,.5,.75,1} passed through actual recording,
measured-state, command construction, and bounded-executor code. State/action
recording flips once; wire/execution preserves native values. Pure modules were
loaded without the driver-loading package initializer; no functions were replaced
and no hardware was accessed. Karma's full pytest collection was unavailable due
to missing optional `trossen_arm`; these standalone assertions cover the relevant
functions without requiring that hardware SDK.

**Deployment limit:** this establishes the local checkout, not the NUC's running
revision or its current gripper calibration. No live robot test was performed.
The user subsequently confirmed that the NUC runs the latest Karma. Upstream
HEAD `4178eefb7d11eb9ae5b9a5ede9864fd65753e26b` was fetched and inspected, and all
25 paired-gripper fixtures passed again. The latest default YAM contract still
uses open=1; its state/action conversion functions support alternate contracts
but default YAM behavior is unchanged. The evidence manifest
[conventions.json](../../outputs/yam-gripper-audit/conventions.json) now has
`verified: true`, records the upstream revision/source hashes, and explicitly
identifies deployment confirmation as user attestation, not a remote NUC check.
Offline preparation was enabled by this audit; no robot commands were issued during the audit.

## Do not confuse the two checkpoints

The existing `/home/sra/molmoact2/scripts/serve_pi05_pink_towel.sh` points to
`outputs/models/yam-pi05-pink-towel-vr`, **not** the original checkpoint above.
That separate fine-tune's saved `README.md:48` and `training_code/upload.py:94`
explicitly state **0=open, 1=closed**. Its HTTP server applies saved processors but
no polarity flip. Directly connecting that server to the inspected Karma native
wire therefore needs an explicit state/input and action/output polarity adapter.
The original checkpoint's `model_open=1` manifest must not be reused for that model.
This is a code-path mismatch finding, not proof of which server/client combination
ran during earlier physical evaluations.

An existing native runtime was also found at
`/home/sra/molmoact2/.venv-pi05-server/bin/python`; the prior statement that no
configured Python 3.12 native environment existed was incomplete. This audit did
not load the full checkpoint or rerun its GPU parity tests.

## Subsequent runtime verification (September 25, 2026)

The native runtime was subsequently used for GPU sampler/processor parity checks,
inversion and training. The completed 34+20-episode run and its final checkpoint
are documented in [BIMANUAL_TRAINING.md](BIMANUAL_TRAINING.md). The statements above
about tests not run describe the original audit date, not the current training
status. Evidence images and conventions JSON remain local, ignored artifacts.
