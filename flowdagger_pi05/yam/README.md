# FlowDAgger for bimanual YAM cloth folding

Native PyTorch/LeRobot adaptation of a frozen pi0.5 policy. A small steering
network predicts the initial sampling noise; the base model generates absolute
joint/gripper action targets. The JAX/openpi MetaWorld reference remains separate.

The current experiment uses 34 successful autonomous episodes and human
corrections from 20 HITL episodes, with no validation/test holdouts, unchanged
recorded frame timing, and the last checkpoint of 50 training epochs.

- [Training configuration, data provenance and completed results](BIMANUAL_TRAINING.md)
- [GPU server and YAM-box rollout/HITL commands](DEPLOYMENT.md)
- [Gripper convention evidence](GRIPPER_AUDIT.md)

## Model contract

| Component | Configuration |
|---|---|
| Base | Frozen `molmoact2-yam-pi05`, a LeRobot pi0.5 checkpoint, not MolmoAct2 architecture |
| Task | `fold the towel` |
| Images | Top, left wrist, right wrist; checkpoint keys top, left, right |
| State/actions | 14 values: left six joints + gripper, right six joints + gripper |
| Actions | Absolute targets; saved quantile processors; no relative-action conversion |
| Sampling | Full 30×32 noise, 10 native Euler steps, RTC/guidance disabled |
| Inversion | Five fixed-point updates per reversed Euler step; float32 latent arithmetic |
| Features | Frozen camera embeddings pooled into 4×4 grids per camera |
| Actor | Shared 64-feature projection, normalized 14-D state, MLP 512/256, linear 960-value output |

All base weights are frozen. Only the projection and steering MLP are optimized.
The 18 unused action channels are completed from a zero-noise base sample's
terminal actions before inversion. Acceptance requires finite values, normalized
RMSE ≤0.03 over all 14 active channels, and each channel RMSE ≤0.05. Physical-unit
errors are stored separately. These are reconstruction gates, not motion guarantees.

Dataset grippers use open=0; the original checkpoint and inspected Karma wire use
open=1. Both state and action grippers are converted before saved preprocessing.
See the audit for evidence and its deployment/revision limits.

## Runtime and local prerequisites

Run commands from the repository root. The configured GPU environment is:

```bash
export PYTHONPATH=/home/sra/molmoact2/third_party/lerobot-pi05-server/src
export OMP_NUM_THREADS=2
/home/sra/molmoact2/.venv-pi05-server/bin/python -m flowdagger_pi05.yam --help
```

Native inference requires the pi0.5-capable LeRobot checkout, its compatible
Transformers/runtime dependencies, Python 3.12 and CUDA. The generic repository
Python 3.11 environment is used for CPU tests, not base-model inference.
`requirements.txt` lists additional backend dependencies; it does not replace
the native runtime's lockfile. Checkpoint/cache identities include model and
processor hashes, tokenizer, native implementation, adapter and runtime versions.
No JAX checkpoint conversion or normalization-statistics recomputation is used.

The convenience training wrapper is configured for this workstation. It requires
existing local human preparation/cache and audit artifacts; a fresh clone is not
sufficient. Exact paths and reconstruction instructions are in the training guide.
Models, videos, caches and local evidence under `outputs/` are not committed.

## Commands and implementation

| File / entry point | Responsibility |
|---|---|
| `python -m flowdagger_pi05.yam` | Individual audit, prepare, invert, train, evaluate, verify and serve commands |
| `run_bimanual.sh` / `run_bimanual.py` | Pinned two-dataset workflow; explicit JSON config overrides generic defaults |
| `snapshots.py`, `snapshots/` | Immutable dataset inventories and hash-verified download/reuse |
| `data.py`, `autonomous_dataset.py` | Camera/frame decoding, control-boundary windows, timing modes, episode assignment |
| `backend.py`, `core.py` | Native prefix/KV-cache sampling, fixed-point inversion and steering actor |
| `pipeline.py`, `preservation.py` | Persistent inversion caches, two-buffer training and optional evaluation |
| `serve.py`, `traces.py` | Karma HTTP inference and explicit execution/outcome trace extension |

The individual `train` command retains its generic defaults. Use the JSON-configured
wrapper for the documented 34+20, last-epoch experiment. Older `run_training.sh`,
`run_tuning.sh` and `run_two_buffer.sh` are historical experiment entry points.

## Collection and method limits

This workflow uses **reconstructed autonomous noise targets** from recorded
commands. The uploaded datasets lack original sampler noise and exact request
boundaries/observations. It is an offline two-buffer approximation, not the full
interleaved on-policy procedure in the [FlowDAgger paper](https://arxiv.org/html/2607.08877v1).

The server logs inference observations, supplied noise, proposed actions, request
IDs, session/generation and actor identity. Full original-noise preservation also
requires client instrumentation: episode IDs, executed indices/commands,
discard/takeover events and final success outcomes. Standard Karma rollout/HITL
commands alone do not establish that all these extension events were delivered.
`POST /events` accepts these events; the client must invalidate stale chunks on
takeover before execution. Freeze the actor for each collection round.

The separate original-trace importer currently requires complete executed chunks
from successful episodes without takeover, matching identities and training
assignments. This is stricter than chunk-level preservation from successful
assisted episodes. It is not the importer used by the current reconstructed run.

## Verification and evidence

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q flowdagger_pi05/yam/tests
```

The suite covers inversion, all-channel gates, camera/gripper contracts, HTTP and
trace handling, temporal gaps, separate dataset provenance, batch balance,
last-epoch saving, and all-training inclusion without holdout evaluation.
Real-checkpoint `verify` checks saved processor round-trip, native sampler parity,
frozen base weights, generated-action inversion and training/serving features.
These checks passed in the completed GPU training workflow.

The user reported better hardware behavior with the earlier validation-selected
actor. No controlled success-rate measurement is recorded here. The final
all-training actor completed offline training; its training MSE does not establish
physical improvement. Former holdout episodes are now training data for that actor;
independent evaluation requires newly collected episodes.
