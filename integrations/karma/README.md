# Karma FlowDAgger collection extension

`flowdagger.patch` targets **SRA-VJTI/karma commit
b4f06f6d645755e605b6c0aec7c10af3d2c911d6**. It was applied and tested in
`/tmp/flowdagger-karma-review`; the actual robot/NUC checkout has not been updated.
`flowdagger_trace.py` is included separately for inspection; the patch already
adds that module to Karma, so no manual copy is required.

On the robot checkout, inspect `git status` and use `git apply --check` before
applying. Do not reset local hardware configuration or overwrite existing changes.

```bash
git rev-parse HEAD
git apply --check /PATH/TO/FlowDAgger/integrations/karma/flowdagger.patch
git apply /PATH/TO/FlowDAgger/integrations/karma/flowdagger.patch
```

Enable tracing explicitly with an absolute, writable journal directory:

```bash
export KARMA_FLOWDAGGER_TRACE_DIR="$PWD/flowdagger-journals/round0"
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

Enter `fold the towel` at each task prompt. The server must be the instrumented
FlowDAgger server, initially without an actor. Startup validates the trace
contract before arm setup. Traced collection requires native 30 Hz, speed 1,
and full 30-step chunks. Controller limits remain active. Ordinary untraced
Karma commands retain their previous recording behavior.

## Changes

- Preserve response request/session/generation/episode metadata alongside each
  action chunk, including prefetch and prefix slicing.
- Deliver `executed` events only after actual dispatch, with post-limit commands.
  Aggregate contiguous execution rows per request; partial chunks are flushed on
  takeover/outcome and cannot become complete autonomous anchors.
- Persist events in an append-only fsynced journal before asynchronous delivery.
  Retry with stable event IDs. The server deduplicates retries. Outcome delivery
  gets a bounded drain after the hardware has parked; delivery failures are logged.
- Immediately invalidate pending policy results on takeover. Handback re-seeds
  from the current state. Worker-thread inference waits leave the command/Quest
  loop responsive. Session, generation and episode checks reject stale results.
- Start the deadline on first episode poll, after setup. Check it before command
  dispatch. Save the episode at expiry. This is a software command deadline, not
  a hard real-time actuator or emergency-stop guarantee.
- Capture recorded camera frames before polling/issuing the corresponding command.
  Record host capture/command timestamps and per-camera host receive/state times.
  Camera reader snapshots atomically return frame, host receive time and sequence.
  Reject stale inference frames. Host receive time is not sensor exposure time
  and does not establish hardware synchronization.
- **For opt-in traced datasets, record grippers directly in the native/model
  convention**, matching the requested identity import. Both state and action
  preserve their values. Quest trigger handling is unchanged. This deliberately
  does not change or relabel legacy datasets.

Keep failed attempts (`n`) as data; distinguish assisted success from autonomous
success. Do not label an unlabeled/discarded/aborted trial successful for anchors.
The strict importer accepts only successful trials without any takeover and only
complete chunks whose commands equal proposals within tolerance.

## Interrupted event delivery

Drain before stopping the GPU server. If the client exits with pending events,
replay its journal **to the original running server session**:

```bash
uv run python -m openpi_control.flowdagger_trace \
  --journal /PATH/TO/session-id.jsonl --server http://GPU_HOST:8204
```

Replay is idempotent. A restarted server has a different session and deliberately
rejects old-session events; do not mix session logs or claim incomplete traces
are valid anchors. Robot control never waits for event HTTP acknowledgements;
new policy inference waits in its worker for earlier events to be acknowledged.

## Tests

The extension tests live in `test_trace.py` and use a fake robot and an in-process
HTTP server. From the FlowDAgger checkout:

```bash
PYTHONPATH=/PATH/TO/karma/src:/PATH/TO/karma/tests:$PWD \
  .venv/bin/python -m pytest -o addopts='' -q integrations/karma/test_trace.py \
  /PATH/TO/karma/tests/test_inference.py /PATH/TO/karma/tests/test_dagger.py \
  /PATH/TO/karma/tests/test_record.py /PATH/TO/karma/tests/test_inference_record.py \
  /PATH/TO/karma/tests/test_cameras.py /PATH/TO/karma/tests/test_cli.py
```

The test recorder confirms both grippers are unchanged in saved traced state and
action, and confirms that images were captured before issuing their commands.
No hardware is opened by these tests.
