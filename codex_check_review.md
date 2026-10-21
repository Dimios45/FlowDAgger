# Review: FlowDAgger for 60-second bimanual YAM towel folding

**Implementation update:** the subsequent user clarification specifies identical
gripper values in recorded data and the model/wire. The new workflow uses
`yam/configs/yam_identity.json` with no import/serve flip. Legacy-source findings
below must not be applied automatically to the user's current datasets. Code,
tests and numerical results are now documented in `flowdagger_pi05/yam_jax/README.md`;
the remainder of this document records the earlier source review.

Reviewed 2026-10-18. This is an implementation and experiment review, not a claim that a new actor has been trained or tested on hardware. The original `codex_check.md` is preserved. No runtime code was changed.

## Decision

Use the specified JAX pi0.5 checkpoint, exact checkpoint preprocessing, one joint bimanual noise actor, and staged on-policy collection. Budget 20 HITL episodes as 10 + 5 + 5, with retraining between rounds. Ten episodes are a first checkpoint, not a promised sufficient dataset. Autonomous evaluation is additional to this correction budget.

The highest-priority changes are execution provenance, observation/time alignment, and held-out model selection. Replacing the MLP with a small DCT head before those fixes would confound the diagnosis.

## What the authors actually established

The paper reports FR3 Duo and Dual UR5e platforms. Bimanual results are Jenga: 0.76→0.86 with 5 additional episodes; toolbox: 0.13→0.80 with 10; plug insertion: 0.60→0.72 with 20. Table 4 labels evaluation as 30 rollouts. Exact camera configurations, teleoperator device, control frequency, chunk/execution horizon, robot-to-task mapping, real-hardware MLP sizes, and round schedules are not specified sufficiently to reproduce them. The method uses correction inversion, noise regression, interleaved aggregation, and equal intervention/autonomous sampling; autonomous targets come from successful autonomous rollouts. [Paper, §§4.1–4.3, 5.5](https://arxiv.org/html/2607.08877v1).

Crucially, the project page labels toolbox as a pi0.5 base trained with **50 demonstrations**, glassware with **30**, and plug insertion with **100**, before adding 10, 5, and 20 corrections respectively. These are not demonstrations that 5–20 episodes teach a fresh generalist every task. The displayed videos run at 3× speed; they cannot establish a one-minute execution requirement. [Author project page](https://microsoft.github.io/FlowDAgger/).

A defensible reconstruction is: start with a task-capable base, run policy-controlled trials, override failures, invert corrective chunks conditioned on their starting observations, aggregate supervision, train the separate actor, and deploy the updated actor. This is a reconstruction of the method, not undisclosed hardware specifics. Towel folding on YAM is our transfer experiment.

## Verified model and observation/action contract

The remote config and model card were fetched directly. Implementation details below were checked against local `expo-ft/expo_ft/conversion/yam_loader.py` and `yam_pi05.py`; pin and compare their hashes with the deployed package before reuse.

| Boundary | Required contract |
|---|---|
| Architecture | pi0.5 action-head model; the repository name does not imply the MolmoAct2 architecture or a world-action latent |
| Cameras | ordered top, left wrist, right wrist; wire names `top_cam`, `left_cam`, `right_cam` |
| JAX image names | `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb` |
| Image preprocessing | RGB uint8 → float [0,1] → source-compatible bilinear aspect-preserving 224×224 letterbox → [-1,1] |
| Physical state/action | `[L j0..5, L grip, R j0..5, R grip]`, width 14; joints in radians |
| Actions | absolute joint/gripper targets, `use_relative_actions=false`; no state subtraction, Cartesian residual conversion, or MetaWorld scaling |
| Normalization | saved state/action quantiles; preserve processor epsilon semantics exactly |
| State conditioning | normalize 14 values, generate source-compatible state-bin prompt, pad continuous state to 32 |
| Prompt | `fold the towel`; do not manually supply the already formatted `Task: ...` template as the instruction |
| Model horizon | 30; internal width 32; noise `(1,30,32)`; action response `(30,14)` |
| Sampling | ten Euler steps; deployment time runs 1→0; no RTC/guidance change during this experiment |

Sources: [checkpoint config](https://huggingface.co/sra-vjti/molmoact2-yam-pi05-jax/blob/main/config.json), [model card](https://huggingface.co/sra-vjti/molmoact2-yam-pi05-jax).

The source LeRobot implementation slices training losses to the physical 14 output dimensions. This does **not** make the other 18 noise channels irrelevant: noisy actions enter a dense projection and can affect physical outputs. Keep all 32 noise channels in the first implementation.

Karma revision inspected: `b4f06f6d645755e605b6c0aec7c10af3d2c911d6`. Its YAM wire uses open=1, while `record.py` converts dataset grippers to open=0. Flip indices **6 and 13** for both recorded state and action when importing these datasets. Do not flip live wire values again. Check the actual robot/client revision and the model's source convention with an open/close test; a config does not independently prove physical polarity. [Karma recording code](https://github.com/SRA-VJTI/karma/blob/b4f06f6d645755e605b6c0aec7c10af3d2c911d6/src/openpi_control/record.py).

## Corrections to codex_check.md

1. **Task-specific base training is missing from the hardware interpretation.** Include the author-page demonstration counts above. Treat the claimed local base success rate as unmeasured until an independent baseline exists.
2. **“Both backends compute the same function” is too strong.** Lossless parameter layout conversion does not establish bitwise execution parity across frameworks, precision settings, kernels, masks, or preprocessing. Test fixed observation + fixed noise in the actual runtime. Pin model revision, processor/tokenizer hashes, OpenPI commit, precision and sampler configuration.
3. **The loader does not expose everything as policy methods.** `inference_functions()` constructs prefix, velocity and sampler functions, but `YamJaxPolicy.__init__` retains only the sampler. Add an explicit backend interface. Its prefix function returns `embed_prefix()` embeddings, not post-transformer prefix hidden states.
4. **The reference inverter is not an exact drop-in paper implementation.** `_build_perstep_fp_reverse_fn` defaults to three iterations and uses an extra reverse-Euler initializer per step. With five refinements it makes 60 velocity evaluations, not 50. The existing YAM `core.invert` initializes from the next state and makes five updates. Choose one version explicitly; verify it against the deployed sampler and time grid.
5. **Do not freeze the unused noise channels as the default.** Completing unused terminal action channels from a base sample is distinct from freezing their initial noise. The proposed 84–140-output actor restricts the inverse solution and must pass a projection/reconstruction ablation first.
6. **Autonomous anchors are not guaranteed to come free.** A corrected successful episode is not a successful autonomous rollout. The existing strict trace importer rejects every episode with a takeover. Reviewed clean segments from assisted episodes are a separate, explicitly labeled approximation; introduce them only with appropriate importer changes and evaluation.
7. **One minute is about 60 full policy chunks, not hundreds.** At 30 Hz there are at most approximately 1,800 control ticks, and fewer complete autonomous chunks when there are interventions, waits or stalls. Human-window overlap does not create independent experience.
8. **Speed 0.5 is not universally repaired by stride-two human sampling.** Time-scaling an autonomous generated chunk and sampling a human trajectory are different operations. Stride two compresses about two physical seconds into one model second. That is an explicit time-warp assumption requiring validation, not recovered ground truth. Use native speed 1.0 after commissioning checks.
9. **Nominal dataset timestamps do not prove real 30 Hz timing.** Recorder frame indices/video PTS can look regular despite inference stalls or repeated camera frames. Log acquisition, command and request times on a monotonic clock.
10. **Do not diagnose the previous failure solely from training MSE or gripper ranges.** Overfitting, action-space mismatch, timing errors, clamp distortion and weak base skill are competing explanations. Small gripper ranges show little motion, not its cause. The old rollout statistics in the plan were not rederived in this review.
11. **Do not require `steered < zero < random` as a universal ranking.** Compare steered against both baselines and multiple random seeds; zero need not beat random. Decoded physical-action error is necessary diagnostic evidence but is not task success.
12. **Thirty evaluation episodes are a pilot, not proof of a ten-point gain.** Near 50% success, one proportion's standard error at n=30 is about 9 percentage points; two independent rates differ with roughly 13-point standard error. Report intervals and counts rather than claiming a small gain is established.

## Required implementation changes, by component

### 1. JAX backend and observation builder

Add `flowdagger_pi05/yam_jax/backend.py` as a thin adapter around the verified processor/model rather than modifying the simulation entry point. Reuse existing YAM data/provenance logic where practical; separate Torch-specific tensor operations from storage schemas.

Expose `prepare`, `encode_features`, `sample`, `velocity`, `normalize_actions`, and `decode_actions`. Cache prefix KV values once per observation for the ten-step sampler and inversion. The local loader's standalone `flow()` recomputes the prefix, which would multiply inversion cost unnecessarily.

Keep the processor's exact quantile denominator: `where(q99 == q01, eps, q99-q01)`. The generic `action_converter.py` adds epsilon unconditionally, zero-fills terminal channels and supports single-final-gripper scaling. It should not be reused unchanged for YAM. Reject invalid shapes/nonfinite values; record out-of-range normalized values without silently changing the model's normalization.

Capture all three camera timestamps and state timestamps. Store the actual server-consumed JPEG bytes or decoded input for autonomous requests. Recorder video frames are not necessarily identical to the prefetched request's observation. Human corrections need a pre-command observation snapshot, not an arbitrary latest image after issuing the command. Add a maximum camera/state age and record repeated frames. Match resolution/framing to the deployed base and compare processed images before declaring camera resolution the failure cause.

### 2. MLP and train pipeline

Start by preserving the current full 960-output actor as a reference while fixing its data and validation. `yam/core.py::SteeringActor` already uses all three spatial camera grids, normalized 14-D state, a shared projection, and 512/256 hidden layers. It is a reasonable baseline; failure has not been isolated to this architecture.

An optional smaller candidate is: ordered per-camera spatial features → shared projection → flatten + normalized state → 256 → 256 → linear 960 → reshape 30×32. Preserve spatial information for towel corners; a single global average of all cameras may lose it. If using prefix hidden features, compute actual transformer outputs, mask padding during pooling, retain camera identity, and verify train/serve parity. For the single fixed prompt, a new language encoder is unnecessary.

Train noise MSE with equal buffer contribution and frozen base weights. Suggested initial engineering settings: AdamW lr=3e-5, weight_decay=1e-3, batch 32 human + 32 autonomous when available, gradient clipping 1, maximum 50 epochs, patience 5 validation evaluations. These are starting settings, not recovered author hardware hyperparameters. Sample across episodes/intervention segments so long or densely overlapped segments cannot dominate. Log human and autonomous losses separately.

Change `yam/configs/yam_cloth_all_34auto_20hitl_50.json` behavior for the new experiment: `all_training=false`, `checkpoint_selection=best`, explicit fixed episode splits. The current all-training/last-epoch configuration is unsuitable for selecting a new actor. Preserve old artifacts as historical results.

Select by held-out decoded normalized and physical errors, including both arms and gripper transitions, plus autonomous preservation. Do not validate neighboring windows from training episodes. Store actor architecture and feature schema with checkpoints; `load_actor` currently hardcodes the existing architecture. Do not hot-swap actors mid-episode or mid-round.

Only try DCT after computing `w_projected = Phi @ (Phi.T @ w)` and measuring decoded-action error on held-out samples. Use N=30 explicitly: `_build_poly_basis` defaults to N=50. Start any compression ablation across all 32 channels (e.g. 10×32 coefficients), and compare to the uncompressed baseline. Smooth actions do not imply smooth initial noise. Gaussian noise retains only K/30 expected temporal energy under a K-term orthonormal projection. Replacing discarded coefficients or the final 18 channels after inversion changes the actions.

### 3. Inversion and human windows

Use 30 consecutive human-controlled commands at actual native cadence, observation at the window start, stride 10 as an initial overlap choice. Never cross takeover, handback, reset, timestamp-gap or episode boundaries; exclude short segments rather than inventing a future by repeating the last command. Include valid corrections from failed episodes too, after checking their local intent.

Initialize terminal 32-D targets from the zero-noise base completion and replace only the first 14 with normalized human commands, as the current YAM pipeline does. This is a reproducible completion choice, not proof it is optimal. Invert the full tensor; do not zero noise channels afterwards. Use exactly the deployed model/precision/time grid. Log fixed-point residuals, finite checks and reconstructed physical errors. The existing normalized RMSE≤0.03 and per-channel≤0.05 gates can remain initial engineering thresholds; calibrate physical tolerances for both joints and grippers.

Separate two tests: (a) invert a full 32-D generated terminal tensor and compare original/recovered noise and decoded actions; (b) reconstruct a 14-D human target with completed terminal channels. Original-noise recovery is not generally implied by physical-action reconstruction alone. Reject poorly reconstructed correction windows and inspect their distribution before spending more hardware episodes.

### 4. Server/client execution traces — blocking integration work

`serve_jax.py` currently calls `predict` without explicit noise and returns actions/timing only. Sample noise explicitly, or invoke the actor; save exact model input, normalized terminal output, physical proposal, noise, task, model/actor/processor identity, request/session/episode IDs and timestamps. Add durable event ingestion with idempotent event IDs and acknowledgements.

In Karma `inference.py::MolmoActClient.infer`, preserve response metadata in a result object; it currently extracts actions and returns only an ndarray. Carry metadata through `ChunkPrefetcher` and `InferenceRolloutSource` alongside each chunk. Log submission observation time, response time, first/last execution time, action indices, interpolation weights, proposal, post-limit actual command and measured state. Record execution after the actual command dispatch in `record.py`, not merely at the earlier policy callback.

Karma's bounded executor modifies commands even at speed 1.0. Existing `yam/traces.py::TraceStore.event` requires proposal/command equality to 1e-6, so it can reject routinely limited chunks. Keep this distinction explicit: exact-original-noise anchors should first use complete, successfully executed, materially unmodified chunks. Log rejected/modified chunks and report eligibility. Do not relabel clipped commands as outputs generated by the logged noise. If nearly all chunks are modified, fix the model/controller contract before loosening acceptance; do not remove limits to obtain cleaner labels.

Instrument takeover, handback, discarded queued/prefetched results, timeouts, episode outcome and reset. Make session/generation checks survive async requests; handback must infer from fresh observations. Ensure outcome success and dataset episode assignment reach the importer. Current `preservation_examples` lacks validation-anchor support and rejects all assisted episodes; extend schema and split assignment deliberately.

Sources: inspected pinned Karma [HTTP/prefetch/executor](https://github.com/SRA-VJTI/karma/blob/b4f06f6d645755e605b6c0aec7c10af3d2c911d6/src/openpi_control/inference.py), [rollout source](https://github.com/SRA-VJTI/karma/blob/b4f06f6d645755e605b6c0aec7c10af3d2c911d6/src/openpi_control/inference_record.py), and [HITL controller](https://github.com/SRA-VJTI/karma/blob/b4f06f6d645755e605b6c0aec7c10af3d2c911d6/src/openpi_control/dagger.py).

## One-minute experiment protocol

Use 60 seconds of wall-clock task time including interventions and inference waits, excluding reset/parking. Record completion time; allow early finish but score success only by a fixed rubric. Suggested rubric: prescribed fold completed, designated edges/corners aligned to a preselected tolerance, towel released and stable on the table. Keep towel type, initial-state distribution, cameras, joint zeros, start pose and controller settings fixed for baseline/comparison.

The current Karma timer checks expiry on control polling. A blocking request can delay that check, and the default request timeout is 60 seconds. For a strict deadline, add an independent deadline guard/cancellable handoff and reject commands arriving after expiry; keep takeover responsive while waiting for inference. Warm up JAX before starting an episode. Do not advertise `--episode-seconds 60` as a hard real-time guarantee by itself.

At 30 Hz and speed 1.0, 30 actions occupy about one second of playback. This does not change the model horizon to 1,800. Start with execute-30; if reaction delay is a measured problem, compare execute-15 under the same baseline and adapted settings. That changes anchor eligibility and doubles the approximate inference rate; full model horizon remains 30. Prefetch reduces waiting but makes observation age important, so log it instead of silently disabling it or adding RTC.

1. Commission the frozen base: left/right order, gripper open/close, camera views, proposal versus command versus measurement, same-noise replay, latency and command limits. Establish that full or near-full towel folding exists in the base's behavior. Use a small diagnostic set first, then retain separate baseline evaluation trials.
2. Round 0: collect 10 HITL trials, each at most 60 seconds. Correct grasp, corner alignment, tension, fold placement and release failures as they occur; hand back when stable. Do not turn all trials into pure teleoperation. Keep failures and local corrections; label final success and assisted status separately.
3. Hold out two whole trials for validation and one for offline testing; seven train initially. If those holdouts lack correction or preservation coverage, redesign the split explicitly before training. Existing evaluation trials can supply training anchors only if formally assigned to training; do not also report them as independent final evaluation.
4. Fit the actor and run a small separate unassisted diagnostic evaluation. If errors concentrate in new on-policy states, collect five more HITL episodes with this frozen actor, aggregate and retrain. Repeat with the final five if useful. Preserve initial holdouts; new collection is training data by default.
5. Aim to reach the target by 20 HITL episodes, but stop and diagnose if inversion fails, successful anchors are absent, or the base never exhibits the needed skill. More MLP epochs cannot repair a convention error or guarantee a new folding skill.
6. Final pilot: 30 base and 30 adapted trials, no human help, matched randomized initial configurations and randomized/interleaved policy order. Report success counts, time-to-success, interventions as failures for autonomous scoring, and uncertainty. Budget these separately from HITL.

Karma accepts the following collection options on the inspected revision (this is a future collection command, not executed in this review):

```bash
uv run karma hitl --rig yam_bimanual \
  --interface left=LEFT_CAN --interface right=RIGHT_CAN \
  --camera-serial top=TOP_SERIAL \
  --camera-serial left_wrist=LEFT_WRIST_SERIAL \
  --camera-serial right_wrist=RIGHT_WRIST_SERIAL \
  --server http://GPU_HOST:8204 --norm-tag yam_dual_molmoact2 \
  --repo-id local/yam-flowdagger-round0 \
  --episodes 10 --episode-seconds 60 --fps 30 \
  --speed 1.0 --chunk-size 30 --num-steps 10 --open-quest
```

Enter `fold the towel` at the HITL task prompt; this subcommand does not use the inference command's `--instruction` flag. The command alone does not provide the proposed trace extension. Karma review choices distinguish keep-success, keep-failure and discard; retain real failures rather than discarding them to inflate success.

## Acceptance checks before new collection

- CPU contracts: ordering, both gripper flips exactly once, quantile conversion, episode/window boundaries, trace joins, late responses after takeover, actual post-clamp commands and deadline handling.
- Real-checkpoint GPU checks: fixed-noise server/offline parity; JAX sampler/inverter round-trip; processor/features parity; finite physical outputs; base weights unchanged. Compare frameworks only if cross-backend use is intended.
- One instrumented trial: replay its exact logged input/noise, join every actual command to the correct request or human segment, reproduce success outcome and buffer eligibility. Confirm the autonomous importer yields usable examples before collecting the rest.
- Training checks: frozen feature schema, no split leakage, balanced buffer updates, best-checkpoint restore, held-out decoded errors and separate preservation metrics.

No GPU numerical parity test, new training run, or hardware evaluation was performed as part of this source review. The recommendations are ready to implement; they are not a validated deployed controller.
