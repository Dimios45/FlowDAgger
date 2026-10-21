# Bimanual YAM: JAX FlowDAgger, 60-second trials

Frozen JAX pi0.5 sampling/inversion with the shared small PyTorch CPU actor and
cache/training/HTTP pipeline. No GPU PyTorch installation is needed for the actor.
This backend does not fine-tune the foundation model or enable RTC.

## Grippers

Use `flowdagger_pi05/yam/configs/yam_identity.json` for the user's confirmed
dataset/model/wire convention. Both grippers pass through unchanged during
import and serving. Quest trigger conversion belongs upstream of the recorded
observation/action contract. Legacy convention manifests remain supported
explicitly; old caches are not silently relabeled. Rebuild inversion caches if
their convention manifest was wrong.

The supplied opt-in Karma patch explicitly records native/model gripper values
in its new datasets. Legacy uninstrumented Karma recording behavior is unchanged.

## Three-camera training path

1. Process `top_cam`, `left_cam`, `right_cam` separately with the checkpoint's
   RGB/224×224 letterbox preprocessing and saved state/prompt processor.
2. The frozen model uses all three images in its prefix attention, plus the task
   and state tokens, to generate actions and invert corrections.
3. For the steering actor, take each camera's **256 projected image tokens of
   width 2048**, before the language transformer. Preserve their 16×16 layout;
   average spatial blocks to a **4×4 grid per camera**.
4. Shared trainable `2048→64` projection; concatenate in top/left/right order:
   `3×16×64=3072` features. Append the normalized 14-D state: **3086 inputs**.
5. `3086→512→256→960`, ReLU hidden layers, linear output; reshape to **30×32
   noise**. The actor has **2,089,728 trainable parameters**. All foundation-model
   weights remain frozen. Both arms are predicted jointly.

These are spatial image features, not pooled transformer hidden states. Training
and serving call the same feature function. There is no new language encoder for
the fixed `fold the towel` prompt, no camera concatenation before image encoding,
and no zeroing of the final 18 noise channels.

## Runtime

Use the checkpoint's pinned `pd-perry/openpi` runtime and its conversion package,
not the simulation submodule. The tested runtime is OpenPI commit
`2abe46282bfdf9f1bc0240f3f9960ec175d1b4a8`. The backend hashes checkpoint parameter
files, processor/tokenizer files and implementation files into cache identity.
It uses float32 by default; changing precision invalidates caches/actors.

On this workstation:

```bash
export PYTHONPATH="$PWD/outputs/yam-jax-review/deps:/home/sra/tirth/expo-ft:/home/sra/tirth/expo-ft/expo_ft/agents/vla/openpi/src:$PWD"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2
YAM_PY=/usr/local/models/sra-expo-ft/venv-jax/bin/python
YAM_CHECKPOINT=/usr/local/models/sra-expo-ft/yam_pi05_jax
YAM_TOKENIZER=/home/sra/molmoact2/outputs/models/paligemma-tokenizer
```

The `outputs/.../deps` directory contains locally installed pandas/dateutil
dependencies; it is not distributed. A fresh environment needs the checkpoint's
JAX serving dependencies plus this repository's YAM preparation/training
dependencies (`torch`, `pandas`, `pyarrow`, `av`, `fastapi`, `json_numpy`, `uvicorn`).

Serve the logged **base** by omitting `--actor`; add `--actor round1.pt` after
training. Freeze the actor for each collection round.

```bash
"$YAM_PY" -m flowdagger_pi05.yam_jax serve \
  --checkpoint "$YAM_CHECKPOINT" --tokenizer "$YAM_TOKENIZER" \
  --conventions flowdagger_pi05/yam/configs/yam_identity.json \
  --traces outputs/round0/traces --normalization-tag yam_dual_molmoact2 \
  --host 0.0.0.0 --port 8204
```

Server startup compiles/warms the JAX sampler before accepting requests. Wait for
`JAX warmup complete` and the Uvicorn readiness message. Warmup does not move arms.

## Prepare, invert, validate and train

Install the [Karma collection patch](../../integrations/karma/README.md).
Collect 10 + 5 + 5 episodes in separate round directories.
Each new round uses the previous actor and the same prompt/controller settings.

For each round, save explicit train/validation/test episode lists in `splits.json`.
Keep all windows of an episode in one split. Preparation currently requires each
split nonempty; five-episode rounds can use 3/1/1. This deliberately reserves new
on-policy validation data as well as the initial holdouts.

```bash
"$YAM_PY" -m flowdagger_pi05.yam_jax prepare \
  --dataset /PATH/TO/round0-dataset \
  --conventions flowdagger_pi05/yam/configs/yam_identity.json \
  --splits splits.json --output outputs/round0/manifest.json

"$YAM_PY" -m flowdagger_pi05.yam_jax invert \
  --checkpoint "$YAM_CHECKPOINT" --tokenizer "$YAM_TOKENIZER" \
  --manifest outputs/round0/manifest.json --output outputs/round0/cache

"$YAM_PY" -m flowdagger_pi05.yam_jax diagnose \
  --checkpoint "$YAM_CHECKPOINT" --tokenizer "$YAM_TOKENIZER" \
  --manifest outputs/round0/manifest.json --window 0 --iteration-sweep \
  --output outputs/round0/inversion-diagnostic.json

"$YAM_PY" -m flowdagger_pi05.yam_jax train \
  --checkpoint "$YAM_CHECKPOINT" --tokenizer "$YAM_TOKENIZER" \
  --cache outputs/round0/cache --autonomous outputs/round0/traces \
  --output outputs/round0/actor.pt
```

Defaults: 50 maximum epochs, patience 5, AdamW 3e-5, weight decay 1e-3, batch 64,
best held-out decoded-action checkpoint. Equal loss contribution and equal batch
counts from interventions and autonomous anchors. Human sampling balances episodes
and their segments; autonomous sampling balances episodes. Original-noise anchors
require complete, unmodified chunks from successful unassisted episodes. A successful
assisted trial is not silently treated as an autonomous success. With no eligible
anchors, collection needs additional successful autonomous examples. Human-only
training is available only as an explicit `--allow-human-only` diagnostic ablation.

For subsequent rounds, make `rounds.json`:

```json
[
  {"cache":"outputs/round0/cache", "traces":"outputs/round0/traces"},
  {"cache":"outputs/round1/cache", "traces":"outputs/round1/traces"}
]
```

```bash
"$YAM_PY" -m flowdagger_pi05.yam_jax merge-caches \
  --sources rounds.json --output outputs/aggregate01/cache
"$YAM_PY" -m flowdagger_pi05.yam_jax train \
  --checkpoint "$YAM_CHECKPOINT" --tokenizer "$YAM_TOKENIZER" \
  --cache outputs/aggregate01/cache --autonomous merged \
  --output outputs/aggregate01/actor.pt
```

Aggregation requires identical base/preprocessing/convention identities, rejects
duplicate datasets, preserves splits, and maps local episode numbers to unique
aggregate IDs while retaining each original observation source. Raw datasets must
remain available. Do not include an old aggregate and its constituent round again.

## What makes inversion relevant, feasible and stable?

**Relevant:** the target must be a genuine 30-frame corrective command sequence,
paired with its pre-command three-camera observation and state. Windows cannot
cross handoff, reset, or timing gaps. Successful reconstruction of a random model
sample proves implementation consistency, not human-correction relevance.

**Feasible:** re-run the frozen sampler with the inferred noise and compare all
14 physical channels to the correction. Store normalized RMSE, each channel's
physical error, left/right arm errors, and both grippers. Initial acceptance gates
remain normalized aggregate RMSE≤0.03 and every channel≤0.05; these are data-quality
gates, not guarantees of grasp/fold success. The inversion cache writes
`inversion-summary.json` with acceptance fraction, RMSE percentiles and per-episode
counts including rejected targets. Inspect grasp, tension, fold and release phases
separately before accepting the dataset as task coverage.

**Stable:** inspect residuals of every inverse Euler step, finite/noise-tail
statistics, the 5/10/20-iteration reconstruction sweep, and finite-difference
amplification. `diagnose` perturbs the normalized action target by RMS 0.001 and
measures change in inferred noise; it also perturbs inferred noise and measures
decoded action changes. Large or erratic amplification warrants investigation.
More iterations need not improve reconstruction. These are local diagnostics,
not a proof of global invertibility or closed-loop stability.

Finally, compare inverted-target reconstruction to zero and multiple random-noise
baselines, then assess the *learned actor* on held-out episodes. Deploying an actor
that improves offline errors still requires separate unassisted hardware trials.
Measure success within 60 wall-clock seconds, time to completion, intervention
count/duration, proposal-versus-command distortion and inference latency.

## Verified during this implementation

- CPU suite covers shared inversion/training/serving, both grippers, physical-channel
  gates, clock gaps, split-safe round aggregation and trace eligibility.
- Patched Karma tests cover recorder, cameras, CLI, inference, HITL/prefetch, event
  delivery and unchanged gripper values in the opt-in traced recorder.
- Real checkpoint on RTX 4090: cached sampler/native max difference **3.28e-7**
  on one recorded towel observation; server predict + wire serialization + logged
  noise replay difference **0**. Base weights unchanged after an actor update.
- Camera ablation: replacing one camera changed only its feature block.
- One recorded **model proposal**, not a human correction: reconstructed normalized
  RMSE **0.007795**, zero-noise **0.202404**, three random seeds **0.203727,
  0.138852, 0.140658**. Target-to-noise amplification **47.73** at perturbation 0.001.
  Five/ten/twenty-iteration RMSE: **0.007795 / 0.008192 / 0.006798**.

These numbers do not certify human correction coverage, closed-loop behavior, or
towel-folding success. No new robot motion or deployable actor training was run.
Detailed local evidence is in `outputs/yam-jax-review/`.
