# Environment Reference

What the task actually simulates, and where each piece lives.

## Registered tasks

| Gym id | Config | Use |
| --- | --- | --- |
| `Isaac-H1-Pathological-Gait-v0` | `H1PathologicalGaitEnvCfg` | Training. Randomized phase, paretic side, pushes and asymmetric domain randomization; spasticity faded in by curriculum. |
| `Isaac-H1-Pathological-Gait-Play-v0` | `H1PathologicalGaitEnvCfg_PLAY` | Evaluation. Fixed start phase, no pushes, no observation noise, spasticity and paretic weakness at full strength. |

## Robot and scene

Unitree H1, 19 actuated joints, from Isaac Lab's `H1_MINIMAL_CFG`. Flat ground plane, whole-body
contact sensing, dome light. `decimation=4` at `sim.dt=0.005` gives a 50 Hz control rate and a
20 s episode (1000 control steps).

PhysX is the default physics backend. A `newton_mjwarp` preset exists but is unvalidated for
this task; select it with `--presets newton_mjwarp`.

## Observation space — 107 dimensions

| Slice | Term | Dim |
| --- | --- | --- |
| `[0:3]` | `projected_gravity` | 3 |
| `[3:6]` | `base_lin_vel` | 3 |
| `[6:9]` | `base_ang_vel` | 3 |
| `[9:28]` | `joint_pos` | 19 |
| `[28:47]` | `joint_vel` | 19 |
| `[47:66]` | `reference_joint_pos` | 19 |
| `[66:85]` | `reference_joint_vel` | 19 |
| `[85:86]` | `paretic_side` (−1 left, +1 right) | 1 |
| `[86:88]` | `gait_phase` as (sin, cos) | 2 |
| `[88:107]` | `last_action` | 19 |

Gait phase is encoded as a sin/cos pair so it stays continuous across the stride wrap.

## Action space — 19 dimensions

The policy does **not** command absolute joint angles. It commands a bounded residual
(`scale=0.25` rad) around the reference pose at the environment's current gait phase, clamped
to the joints' soft limits. That keeps the search space near clinically plausible gait from
the first iteration.

The same action term injects the paretic limb's TSRT reflex torque as a feed-forward joint
effort. H1's joints use implicit (in-simulation PD) actuators, and Isaac Lab passes
`joint_efforts` through to the solver alongside the position target, so the reflex **adds to**
the actuator torque rather than replacing it. The policy has to work against the spasticity,
which is the point of the model.

## The pathology

Three deficits, each grounded in the clinical literature rather than hand-tuned difficulty.

### TSRT spasticity — `tsrt.py`

Tonic Stretch Reflex Threshold (Feldman 1986; Levin & Feldman 1994; Musampa et al. 2007). A
spastic muscle recruits when the joint is stretched past a threshold angle, and that threshold
falls as stretch velocity rises — which is why spasticity is velocity-dependent:

```
θ_th = λ₀ − μ · θ̇⁺
τ    = −( k · max(θ − θ_th, 0) + b · θ̇⁺ )
```

Modelled on three joints of the paretic leg only:

| Joint | Clinical presentation |
| --- | --- |
| ankle | Plantarflexor spasticity resisting dorsiflexion — **foot drop**. |
| knee | Resists flexion — **stiff-knee gait**. |
| hip roll | Adductor spasticity resisting abduction — **scissoring / circumduction**. |

### Hemiparetic weakness — `mdp/events.py`

The paretic leg's actuator effort ceiling is cut to 40 % of nominal on reset, putting H1's
300 Nm leg actuators near the 120 Nm figure used clinically, and its 100 Nm ankle near 40 Nm.

### Asymmetric domain randomization — `mdp/events.py`

Randomization here is deliberately **asymmetric**. Perturbing the paretic limb as widely as the
sound one would wash out the fragile pathological limit cycle — the deficit would just look
like noise. So:

| Property | Paretic limb | Sound limb |
| --- | --- | --- |
| joint stiffness | ±2 % | ±15 % |
| joint damping | ±2 % | ±30 % |
| link mass | ±2 % | −20 % / +25 % |
| effort limit | 40 % of nominal | ±15 % |

Balance pushes are also asymmetric: a perturbation toward the paretic side is scaled by 1.45×,
because post-stroke balance fails asymmetrically and the recovery burden belongs where the
clinical deficit is.

## Rewards

| Term | Weight | Notes |
| --- | --- | --- |
| `joint_pos_tracking` | 15.0 | Clinically weighted RBF. Ankles ×3.0, knees ×2.5, hips ×1.0–2.0, arms ×0.5. |
| `joint_vel_tracking` | 2.0 | Same weighting. |
| `base_height` | 3.0 | Pelvis near 1.05 m. |
| `forward_velocity` | 2.0 | Target 0.5 m/s in the robot frame. |
| `alive` | 2.0 | |
| `margin_of_stability` | 1.5 | See below. |
| `paretic_foot_clearance` | 1.0 | Swing-phase only; shapes foot drop without fighting stance. |
| `flat_orientation_l2` | −2.0 | |
| `dof_pos_limits` | −1.0 | |
| `action_rate_l2` | −0.01 | |
| `dof_torques_l2` | −1e−5 | |
| `dof_acc_l2` | −1.25e−7 | |
| `termination_penalty` | −200.0 | |

### Margin of stability

The Extrapolated Centre of Mass and mediolateral Margin of Stability of Hof et al. (2005):

```
XCoM = CoM + CoM_velocity / ω₀,    ω₀ = √(g / CoM_height)
MoS  = distance from XCoM to the edge of the support polygon
```

Unlike the analytical predecessor this replaces, which approximated foot placement from the
root pose and assumed permanent double support, this reads the **real** mass-weighted whole-body
centre of mass and builds the support polygon from the feet **actually in contact this step**,
via contact sensors. Everything is expressed in the robot's yaw frame, so "lateral" stays
lateral however the robot is heading.

A negative margin means the XCoM has left the support polygon — the robot is committed to a
fall it cannot arrest without stepping — and is penalised quadratically.

## Terminations

| Term | Condition |
| --- | --- |
| `time_out` | 20 s episode cap. |
| `base_height_fall` | Pelvis below 0.65 m. |
| `bad_orientation` | Tilt beyond 1.0 rad. |
| `torso_contact` | Illegal contact on the torso. |
| `tracking_divergence` | Weighted RMS joint error above 1.0 rad — the rollout has stopped carrying learning signal. |

## Curricula

Both are **step-based, not iteration-based**, so changing `--envs` changes how many iterations
they span. See the training guide.

| Term | Ramp |
| --- | --- |
| `spasticity` | TSRT reflex gain 0 → 1 over 24 000 steps. |
| `push_magnitude` | Push velocity 0 → configured over 24 000 steps. |

Both are disabled in the play task, where the deficits run at full strength throughout.

## Module map

```
tasks/humanoid_pathological_gait/
├── assets.py               Clinical data file resolution
├── h1_joints.py            Clinical ↔ simulation joint ordering, mirroring, tracking weights
├── reference.py            Reference stride playback, per-env gait phase and paretic side
├── tsrt.py                 TSRT spasticity model
├── h1_pathological_env.py  ManagerBasedRLEnv subclass; owns gait state and AMP accessors
├── mdp/
│   ├── actions.py          Reference-residual action term + reflex torque injection
│   ├── observations.py     Reference kinematics, paretic side, gait phase
│   ├── rewards.py          Tracking kernels, XCoM/MoS, foot clearance
│   ├── terminations.py     Reference tracking divergence
│   ├── events.py           Reference-pose resets, asymmetric randomization, paretic pushes
│   └── curriculums.py      Spasticity and push ramps
└── config/h1_pathological/ Env config, gym registration, RSL-RL agent config
```

`algorithms/` holds the PPO and AMP implementations. They are plain PyTorch with no Isaac Lab
dependency, so they can be exercised on CPU without booting a simulator.
