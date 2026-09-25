# Serving and collecting on bimanual YAM

## GPU server (omen)

Stop any previous server using port 8206. Run from the FlowDAgger checkout:

```bash
cd /home/sra/tirth/FlowDAgger
export PYTHONPATH=/home/sra/molmoact2/third_party/lerobot-pi05-server/src
export OMP_NUM_THREADS=2

/home/sra/molmoact2/.venv-pi05-server/bin/python -u \
  -m flowdagger_pi05.yam serve \
  --checkpoint /home/sra/molmoact2/outputs/models/molmoact2-yam-pi05 \
  --tokenizer /home/sra/molmoact2/outputs/models/paligemma-tokenizer \
  --actor outputs/yam-flowdagger-v2/training/bimanual-20260925T123943Z-lTrWYA/actor.pt \
  --conventions outputs/yam-gripper-audit/conventions.json \
  --normalization-tag yam_dual_molmoact2 \
  --traces outputs/yam-flowdagger-v2/live-traces-all-training-epoch49 \
  --device cuda \
  --host 0.0.0.0 \
  --port 8206
```

This is one combined server: steering actor → initial noise → frozen pi0.5 sampler
→ saved action postprocessor → absolute YAM targets. No separate base server is
needed for steered inference. Switching actors requires restarting this process;
training a new checkpoint does not replace an actor already loaded in memory.

In another terminal:

```bash
curl -fsS http://localhost:8206/act
```

Health reports checkpoint hashes, actor version, normalization tag, three cameras,
14-D state/action, 30-action horizon and ten sampler steps. Requests with other
sampler step counts, incompatible normalization tags/instructions or RTC/guidance
fields are rejected. `/act` supports JSON NumPy JPEG arrays or raw RGB images.

## YAM-box client

Run in the **Karma checkout on the robot client**. The addresses, CAN interfaces
and serials below match this installation; change them if hardware/network mapping
changes. Keep each continuation backslash as the last character on its line.
Karma uses the `yam_dual_molmoact2` normalization contract. These commands execute
real robot motion and record episodes locally.

### Autonomous rollout: 10 episodes

```bash
uv run karma rollout \
  --interface left=can_left \
  --interface right=can_right \
  --camera-serial top=348523020354 \
  --camera-serial left_wrist=254623070863 \
  --camera-serial right_wrist=254623070417 \
  --server http://192.168.0.167:8206 \
  --repo-id local/yam-flowdagger-epoch49-rollout-01 \
  --episodes 10 \
  --episode-seconds 300 \
  --fps 30 \
  --speed 0.5 \
  --chunk-size 30 \
  --num-steps 10 \
  --no-prefetch \
  --no-cuda-graph
```

### Human-in-the-loop collection: 3 episodes

```bash
uv run karma hitl \
  --interface left=can_left \
  --interface right=can_right \
  --camera-serial top=348523020354 \
  --camera-serial left_wrist=254623070863 \
  --camera-serial right_wrist=254623070417 \
  --server http://192.168.0.167:8206 \
  --repo-id local/yam-flowdagger-epoch49-hitl-01 \
  --episodes 3 \
  --episode-seconds 300 \
  --fps 30 \
  --speed 0.5 \
  --chunk-size 30 \
  --num-steps 10 \
  --no-prefetch \
  --no-cuda-graph
```

At each episode prompt enter exactly **`fold the towel`**. Use a new dataset ID
for each new recording session. Follow the client's displayed takeover/handback
controls and label the actual episode outcome. These commands do not automatically
aggregate the new data into the training snapshot.

Speed is a client playback setting: 30 model actions become 59 control ticks at
0.5. The final actor was trained on consecutive recorded commands without undoing
previous playback interpolation. This setting therefore stretches those predicted
sequences again; keep it explicit when comparing actors. Prefetch is disabled in
these examples. Server timing alone does not capture request-to-execution delay.

## Trace and evaluation limits

Server trace files contain the observation/noise/proposed action for each request.
Without client-supplied episode IDs and `/events` execution/outcome reporting,
requests may be `unassigned` and cannot be treated as verified successful
original-noise preservation data. See the [backend guide](README.md#collection-and-method-limits).

Compare the final actor against the previous actor and ordinary base inference
under matched starting conditions and client settings. Record autonomous success,
interventions, completion time and gripper failures. The all-training actor has no
held-out offline score; its final noise MSE is not a folding-success metric.
