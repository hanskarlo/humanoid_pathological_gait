# Training Guide

Everything needed to take this extension from a fresh clone to a trained post-stroke gait
policy, and to tell a good run from a bad one.

- [1. Prerequisites](#1-prerequisites)
- [2. First-time setup](#2-first-time-setup)
- [3. Preflight verification](#3-preflight-verification)
- [4. Training](#4-training)
- [5. Reading the training output](#5-reading-the-training-output)
- [6. Playback and evaluation](#6-playback-and-evaluation)
- [7. Tuning](#7-tuning)
- [8. Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

| Requirement | Notes |
| --- | --- |
| NVIDIA GPU | Required. Developed on an RTX 5060 Ti (16 GB). VRAM caps `--envs`; see [§7](#71-how-many-environments). |
| Linux | Isaac Sim 6.0 headless. Windows is untested here. |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | Resolves the environment, including Isaac Sim (~25 GB on a cold cache). |
| Internet, first run only | The H1 USD is fetched from `ISAACLAB_NUCLEUS_DIR` and cached. |
| ~40 GB free disk | Isaac Sim, kernel caches, and checkpoints. |

### The Omniverse EULA

Isaac Sim prompts for the NVIDIA Omniverse licence agreement on first launch and **hangs on
that prompt under any headless or scripted run**. Accept it non-interactively:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

The shell scripts set this for you if it is unset, but they print a notice when they do —
setting a licence-acceptance variable is your decision, not the tooling's. Put the export in
your shell profile to silence it. The agreement is at
<https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html>.

---

## 2. First-time setup

```bash
cd humanoid_pathological_gait
scripts/setup.sh
```

This resolves the environment with `uv sync`, checks that Isaac Lab imports and a CUDA
device is visible, reports whether the clinical data files are staged, and boots Isaac Sim
once headless to warm the kernel and asset caches. **The first boot takes several minutes**
while Warp compiles kernels; later boots take ~10 s.

`scripts/setup.sh --skip-sync` re-runs the checks without touching dependencies.

### Clinical data

Two files drive the task. See [data.md](data.md) for schemas and regeneration.

| File | Role | Required? |
| --- | --- | --- |
| `retargeted_h1_stride.npz` | The reference stride the policy tracks and resets onto. | **Yes.** Committed to this repo. |
| `stroke_gait_dataset.npz` | The AMP expert motion corpus. | No, but strongly recommended. Git-ignored at 31 MB. |

Without the corpus, the AMP discriminator falls back to the single reference stride and then
to a synthetic prior. Training still runs, but the adversarial term is far weaker and the
resulting gait is correspondingly less faithful to real post-stroke kinematics. `setup.sh`
tells you which case you are in.

---

## 3. Preflight verification

A full training run is long. This gate catches, in about three minutes, most of what would
otherwise fail an hour in:

```bash
scripts/verify.sh          # full gate
scripts/verify.sh --quick  # skip the RSL-RL learning check
```

| Check | What a failure means |
| --- | --- |
| task registration | The gym ids are not registered — usually a broken editable install. |
| zero-action rollout, training + play tasks | Non-finite observations or rewards. Never start a run on a NaN. |
| pathology machinery is live | The thing that makes this task *pathological* is not actually engaged. See below. |
| environment drives a standard learner | 5 iterations of stock RSL-RL PPO. Catches shape and manager errors independently of the AMP code. |

The pathology check asserts eleven properties that only mean something against real physics:
the TSRT reflex torque engages on the paretic limb and **never** leaks to the sound one, both
feet register contact, the margin of stability is finite and physically scaled, resets land
on the reference pose, the paretic side is sampled on both sides, and the asymmetric
randomization actually reached the solver with the paretic limb held tight. Run it directly
for the itemised report:

```bash
uv run python scripts/check_pathology.py --headless
```

---

## 4. Training

```bash
scripts/train.sh                       # full run: 4096 envs, 3500 iterations
scripts/train.sh --preset short        # 512 envs, 500 iterations
scripts/train.sh --preset smoke        # 64 envs, 20 iterations (~1 min)
scripts/train.sh --envs 2048 --iters 5000 --seed 7
```

Presets are starting points, not tuned recipes:

| Preset | Envs | Iterations | Purpose |
| --- | --- | --- | --- |
| `smoke` | 64 | 20 | Proves the loop runs. Not a policy. |
| `short` | 512 | 500 | Shows the shape of the learning curve. |
| `full` | 4096 | 3500 | Publication run. |

Each run writes to `logs/ppo_amp/<timestamp>/`, containing `train.log` and `model_<iter>.pt`
checkpoints. Anything after `--` is forwarded to the Python script:

```bash
scripts/train.sh --preset short -- --amp_weight 8.0 --lr_disc 5e-5
```

### What is actually being optimised

The environment's reward manager computes kinematic tracking, margin of stability, foot
clearance and the whole-body regularizers. The training loop adds one term on top:

```
r_total = r_env + amp_weight * r_amp
```

`r_amp` is the adversarial style reward — how well the motion passes as real post-stroke gait
to a discriminator trained against the clinical corpus. It is the reason this loop exists
rather than stock RSL-RL, which has no AMP support in this Isaac Lab release.

### The RSL-RL path

```bash
scripts/train.sh --rsl-rl --preset smoke
```

This is an **environment sanity check, not a training route**. Without the motion prior the
policy optimises reference tracking alone, so it will not reproduce the qualitative gait
signature. Use it to isolate whether a problem is in the environment or in the AMP code.

---

## 5. Reading the training output

```
[iter 00120/3500 |   102.4s] env_r    3.812 | amp_r  0.641 | ep_len  186.3 | pol  -0.0231 | val   88.417 | disc  0.9042
```

| Column | Healthy behaviour |
| --- | --- |
| `env_r` | Rises steadily. This is the environment reward, dominated by joint tracking. |
| `amp_r` | Settles around 0.4–0.8. **Not** a quantity to maximise — see below. |
| `ep_len` | Rises toward the episode cap (20 s ÷ 0.02 s = 1000 steps). The clearest single health signal: a policy that stops falling is learning. |
| `pol` | Small and negative. Magnitude spikes mean the trust region is being strained. |
| `val` | Falls from a large initial value as the critic calibrates. |
| `disc` | Falls off its initial value, then plateaus. |

**On `amp_r`:** this is adversarial, so it is not monotone and should not be. If it climbs to
~1.0 and stays there, the discriminator has been beaten and has stopped providing signal
(lower `--lr_disc`, or raise the gradient penalty). If it collapses toward 0, the
discriminator is winning outright and the style reward has become noise (raise `--lr_disc`, or
check that the expert corpus is actually staged). A healthy run keeps both in tension.

### The curricula

Two deficits fade in over the first 24 000 environment steps rather than starting at full
strength, because a policy that begins on the floor never learns to walk:

- **spasticity** — the TSRT reflex gain ramps 0 → 1.
- **push magnitude** — balance perturbations ramp 0 → configured strength.

They are step-based, not iteration-based, so **changing `--envs` changes how many iterations
the ramp spans**. At 4096 envs the ramp is ~6 iterations; at 64 it is ~375. If you train at a
small `--envs` for a long run, lengthen `num_steps` on those curriculum terms to match.

Playback evaluation always runs both at full strength — they are the phenomenon under study,
not a training aid.

---

## 6. Playback and evaluation

```bash
scripts/play.sh                                   # most recent checkpoint
scripts/play.sh --checkpoint logs/ppo_amp/.../model_3500.pt
scripts/play.sh --video                           # record to <checkpoint dir>/videos/
scripts/play.sh --gui                             # Isaac Sim viewport
```

Playback uses the deterministic `Isaac-H1-Pathological-Gait-Play-v0` task — fixed start phase,
no pushes, no observation noise — and the policy **mean** action rather than a sample, so
repeated runs of one checkpoint are comparable.

```
  weighted RMS joint tracking error :   0.1075 rad
  forward speed                     :   0.4821 m/s
  mediolateral margin of stability  :   0.0412 m
  peak paretic reflex torque        :   5.568 Nm
  mean episode length               :    873.2 steps
```

| Metric | What to look for |
| --- | --- |
| tracking error | Lower is closer to the clinical reference. |
| forward speed | Should be **positive** and near the 0.5 m/s target. Negative means the policy learned to walk backwards — a real failure mode of undertrained checkpoints. |
| margin of stability | Positive means the extrapolated centre of mass sits inside the support polygon. Persistently negative means it is falling and catching itself. |
| peak reflex torque | Non-zero confirms spasticity is engaged. Zero means the curriculum never ramped. |
| episode length | Near the 1000-step cap means it is not falling. |

---

## 7. Tuning

### 7.1 How many environments

`--envs` is bounded by VRAM. 4096 is comfortable on 16 GB for this scene. If you hit CUDA
OOM, halve it. Fewer environments means noisier gradients, so consider raising
`--num_steps_per_env` to keep the batch size up.

### 7.2 Knobs worth turning

| Flag | Default | Effect |
| --- | --- | --- |
| `--amp_weight` | 5.0 | Style vs. task. Higher tracks the clinical *character* of the gait more closely at some cost to the explicit reward terms. |
| `--lr_disc` | 1e-4 | Discriminator learning rate. The main lever on the adversarial balance described in [§5](#5-reading-the-training-output). |
| `--lr_policy` | 3e-4 | Policy/value learning rate. |
| `--num_steps_per_env` | 24 | Rollout length per iteration. Raise for less noisy advantage estimates. |

Reward weights, curriculum lengths, TSRT parameters and randomization ranges live in the
environment config, not on the command line:
`source/humanoid_pathological_gait/humanoid_pathological_gait/tasks/humanoid_pathological_gait/config/h1_pathological/h1_pathological_env_cfg.py`.

### 7.3 Physics backend

PhysX is the default and the only backend validated here. A Newton/MJWarp preset exists:

```bash
scripts/train.sh --preset smoke -- --presets newton_mjwarp
```

The task leans on contact forces and feed-forward joint efforts; both are better-tested under
PhysX. Treat Newton as experimental and re-run `scripts/verify.sh` after switching.

---

## 8. Troubleshooting

### Isaac Sim hangs with no output

The EULA prompt. See [§1](#the-omniverse-eula).

### `RuntimeError: Caught an unknown exception!` from `app.startup`

A script imported `isaaclab.envs.ManagerBasedRLEnv` — which loading any task config
transitively does — **before** `AppLauncher` started Kit. On this Isaac Sim build that leaves
the USD/`pxr` bindings in a state Kit cannot start from.

Every script here launches the simulation app first and imports task modules after. Follow
that pattern in new scripts. This also rules out `launch_simulation` from
`isaaclab_tasks.utils`, which requires the config up front.

### `uv run isaaclab zero_agent` — no such command

The `isaaclab` console script installed with Isaac Lab 3.0.0b2 only dispatches `train` and
`play`, and those point at a `scripts/reinforcement_learning/` directory the wheel does not
ship. Use this extension's scripts instead.

### CUDA out of memory

Lower `--envs`. See [§7.1](#71-how-many-environments).

### `amp_r` pinned at 1.0, or collapsed to 0

The discriminator has lost or won outright. See [§5](#5-reading-the-training-output).

### Reflex torque is zero in playback

The spasticity curriculum never ramped, most likely because the run was too short at a high
`--envs`. See the curriculum note in [§5](#the-curricula). Playback of the `-Play-v0` task
holds spasticity at full strength regardless, so a zero here points at the checkpoint's
training conditions rather than at playback.

### Policy walks backwards

Normal for undertrained checkpoints — negative `forward speed` in playback. The forward
velocity reward is one term among many and takes a while to dominate. If it persists past a
few hundred iterations at `--preset short`, raise the `forward_velocity` reward weight in the
environment config.
