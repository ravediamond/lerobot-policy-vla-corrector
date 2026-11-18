# lerobot_policy_vla_corrector

VLA-Corrector ([arXiv 2607.01804](https://arxiv.org/abs/2607.01804), code
[ZJU-OmniAI/vla-corrector](https://github.com/ZJU-OmniAI/vla-corrector), Apache-2.0) as a policy plugin for
[LeRobot](https://github.com/huggingface/lerobot) 0.7.0, policy type `vla_corrector`.

It wraps a frozen flow-matching policy (SmolVLA in 0.1.0) and adds two inference-time parts, without retraining the
base:

- a **latent vision monitor**: a residual MLP (4 × 2048; 38.1M parameters on SmolVLA's 960-wide vision tokens)
  predicts how the base policy's own vision tokens change k steps after an action. When the observed change keeps
  disagreeing with the prediction (cosine mismatch above a median + MAD band for `patience` steps), the action queue
  is truncated and the base re-plans;
- **online gradient guidance (OGG)**: in the re-planned chunk, every third Euler step of the flow-matching denoiser
  replaces the velocity `v` by `v − η ∇_v L`, where `L` is the cosine loss between the monitor's prediction for the
  first action and the correction target.

Only the corrector is vendored (rewritten, upstream commit `927a06d`, see `NOTICE`). Base policies come from
`lerobot==0.7.0` unchanged.

## Install

```
pip install "lerobot[smolvla]==0.7.0"
pip install git+https://github.com/ravediamond/lerobot-policy-vla-corrector@v0.1.0
```

LeRobot registers packages named `lerobot_policy_*` at start-up, so `--policy.type=vla_corrector` then works in the
LeRobot CLIs.

## Modes

| `mode` | what runs |
|---|---|
| `off` | the base policy only; outputs are bit-identical to the base (tested) |
| `monitor_only` | monitor runs and logs, actions are the base's |
| `truncate` | on a trigger the queue is cleared and the base re-plans |
| `truncate_ogg` | as `truncate`, and the re-plan is guided by OGG (`ogg_calls_after_trigger` re-plans, default 1) |

Every `select_action` call leaves its telemetry in `policy.last_info` (and `policy.telemetry`): mismatch score,
threshold, release level, triggered, truncated queue length, re-planned, OGG calls, OGG loss and gradient norm,
monitor ms, step ms.

Defaults follow the paper and the upstream code: k_step 10, window 15, λ_on 3.0, λ_off 2.0, patience 5, cooldown 10,
η 1.0, guidance on every third denoise step, one guided re-plan per trigger. `monitor_every_n_steps` (default 1)
runs the monitor on every n-th control step; `k_step` must be a multiple of it.

## Train a corrector

Upstream releases no corrector weights; train one per base policy and task data.

```
python scripts/extract_latents.py --base <your SmolVLA checkpoint> --repo-id <dataset> \
    --rename observation.images.front=observation.images.camera1 --dtype bf16 --out latents/
python scripts/train_corrector.py --latents latents/ --base <your SmolVLA checkpoint> --epochs 30 --out corrector/
```

`extract_latents.py` writes one shard per episode, skips shards already written and stops below `--min-free-gb`, so a
large dataset can be processed a slice at a time. Actions are stored in the base policy's normalised space, which is
what the monitor sees at run time. `lerobot-train --policy.type=vla_corrector` also works (`forward` returns the
corrector loss on frame pairs t, t + k_step) but encodes both frames at every step. A corrector trained with the
upstream code loads with `lerobot_policy_vla_corrector.upstream_checkpoint.load_upstream_corrector`.

## Use

```python
from lerobot_policy_vla_corrector import VLACorrectorConfig, VLACorrectorPolicy

cfg = VLACorrectorConfig(base_policy_path="<SmolVLA checkpoint>", corrector_path="corrector/", mode="truncate_ogg")
policy = VLACorrectorPolicy(cfg)
policy.reset()
action = policy.select_action(batch)
```

The base policy's pre- and post-processors are used unchanged (`make_vla_corrector_pre_post_processors`).

## Parity with the upstream code

`scripts/parity_upstream.py` (run with the upstream fork on `PYTHONPATH`) and `scripts/parity.py` compare the plugin
with upstream on one corrector trained by upstream's own scripts (2 epochs on 10 episodes of
`lerobot/svla_so101_pickplace`, `lerobot/smolvla_base`, fp32, Jetson Orin NX):

| check | result |
|---|---|
| mismatch scores, 500 steps of 5 episodes, max \|Δ\| | 6.36e-07 |
| trigger timesteps | identical (4 triggers) |
| OGG-guided chunk with the same noise, max \|Δ\| | 0.0 |
| `mode=off` vs the base policy, 200 steps, max \|Δ\| | 0.0 |


## What we found before this port

An earlier test on our SO-101 bowls task used the monitor alone, read offline: trained on 157 demonstrations and
replayed on 50 rollouts with the paper's trigger, it flagged all 8 failures and 40 of 42 successes, about 10
interrupts per minute on successful runs, with an episode-level AUROC of 0.72 against 0.67 when the monitor was fed
another episode's actions. As a failure detector it failed. That test used mean-pooled latents at 15 Hz on one task
and did not truncate or re-plan, which is what the paper does with the trigger. This plugin exists to test that use,
with the false-trigger rate on clean runs reported next to every success rate.

## Limits of 0.1.0

- SmolVLA backend only; Pi0.5 is next.
- Local inference only. With `lerobot-policy-server` the truncation would have to happen in the client's queue.
- The monitor runs with batch size 1; batched inference is refused unless `mode=off`.
- A base policy with RTC enabled is refused.
- No corrector weights are published yet and no benchmark numbers are claimed here.

## License

Apache-2.0. The corrector is derived from ZJU-OmniAI/vla-corrector (Apache-2.0); see `NOTICE`.
