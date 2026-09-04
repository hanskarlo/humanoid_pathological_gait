# humanoid_pathological_gait

An [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) extension that simulates **post-stroke
hemiparetic gait** on the 19-DoF Unitree H1 humanoid, and trains policies that reproduce it.

The pathology is simulated, not scripted. Spasticity is a velocity-dependent reflex torque the
PhysX solver integrates; balance is measured from the real whole-body centre of mass against a
support polygon built from contact sensors; the paretic limb carries a reduced torque ceiling
and is deliberately shielded from the domain randomization that hardens the sound limb. The
reference motion is real clinical capture, retargeted onto H1.

```bash
scripts/setup.sh     # resolve the environment, warm caches
scripts/verify.sh    # preflight gate
scripts/train.sh     # PPO + Adversarial Motion Prior
scripts/play.sh      # play back the newest checkpoint with gait metrics
```

## Documentation

| Guide | Contents |
| --- | --- |
| [docs/training.md](docs/training.md) | Setup, verification, training, reading the output, tuning, troubleshooting. **Start here.** |
| [docs/environment.md](docs/environment.md) | Observation/action spaces, the pathology models, rewards, terminations, module map. |
| [docs/data.md](docs/data.md) | Clinical data schemas, joint ordering, regeneration, ethics. |

## Registered tasks

| Gym id | Purpose |
| --- | --- |
| `Isaac-H1-Pathological-Gait-v0` | Training. Randomized phase, side, pushes and asymmetric domain randomization; spasticity faded in by curriculum. |
| `Isaac-H1-Pathological-Gait-Play-v0` | Evaluation. Deterministic: fixed start phase, no pushes, no observation noise, spasticity at full strength. |

## Requirements

An NVIDIA GPU, Linux, and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Isaac Sim (~25 GB) is resolved as a dependency. Isaac Sim also requires accepting the NVIDIA
Omniverse licence agreement:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

Without it, every headless run hangs on an interactive prompt. The shell scripts will set it
for you and say so, but it is your acceptance to give — see
[docs/training.md](docs/training.md#the-omniverse-eula).

## Installation

```bash
git clone git@github.com:hanskarlo/humanoid_pathological_gait.git
cd humanoid_pathological_gait
scripts/setup.sh
```

`setup.sh` runs `uv sync`, checks that Isaac Lab imports and a CUDA device is visible, reports
whether the clinical data is staged, and boots Isaac Sim once to warm the kernel and asset
caches. The first boot takes several minutes; later ones take about ten seconds.

## Scripts

| Script | Purpose |
| --- | --- |
| `scripts/setup.sh` | One-time environment setup and cache warm-up. |
| `scripts/verify.sh` | Preflight gate: registration, NaN-free rollouts, pathology assertions, a 5-iteration learning check. |
| `scripts/train.sh` | PPO+AMP training. Presets `smoke` / `short` / `full`. |
| `scripts/play.sh` | Playback with clinical gait metrics; optional video. |
| `scripts/list_envs.py` | Registered tasks and their physics presets. |
| `scripts/zero_agent.py` | Zero-action rollout; checks observations and rewards for NaNs. |
| `scripts/check_pathology.py` | Eleven assertions that the pathology machinery is live. |
| `scripts/train_amp.py` | The PPO+AMP loop itself. |
| `scripts/train_rsl_rl.py` | Stock RSL-RL PPO — environment sanity check only, no AMP support. |
| `scripts/play.py` | The playback loop itself. |

## Two things that will bite you

**The `isaaclab` console script cannot run these tasks.** The one installed with Isaac Lab
3.0.0b2 only dispatches `train` and `play`, and those point at a `scripts/reinforcement_learning/`
directory the wheel does not ship. There is no `zero_agent`, `random_agent` or `benchmark`
subcommand. Use this extension's scripts.

**Import order matters.** On this Isaac Sim build, importing `isaaclab.envs.ManagerBasedRLEnv`
— which loading any task config transitively does — before `AppLauncher` starts Kit leaves the
USD/`pxr` bindings in a state Kit cannot start from, and `app.startup` dies with "Caught an
unknown exception!". Every script here launches the simulation app first and imports task
modules after. Follow that pattern in new scripts; `launch_simulation` from
`isaaclab_tasks.utils`, which wants the config up front, cannot be used.

## Development

```bash
uv run pre-commit run --all-files                  # format and lint
uv run python .vscode/tools/setup_vscode.py        # VS Code / Pylance paths
```

Add the `source/` directory to the Isaac Sim Extension Manager search paths and enable the
extension under *Third Party* to use it inside the Isaac Sim GUI.

## Citation

The models implemented here follow:

- Feldman (1986); Levin & Feldman (1994); Musampa et al. (2007) — Tonic Stretch Reflex Threshold.
- Hof, Gazendam & Sinke (2005), *The condition for dynamic stability* — Extrapolated Centre of
  Mass and Margin of Stability.
- Peng et al. (2021), *AMP: Adversarial Motion Priors* — the adversarial style reward.

## License

BSD 3-Clause. See [LICENSE](LICENSE). The clinical gait corpus is **not** covered by this
licence and is not distributed with the code; see [docs/data.md](docs/data.md#ethics-and-licensing).
