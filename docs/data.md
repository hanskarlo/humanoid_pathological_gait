# Clinical Data

The task is driven by real post-stroke motion capture, not synthetic gait. Two files carry
that data into the simulation.

## Files

Both are resolved by
`source/humanoid_pathological_gait/humanoid_pathological_gait/tasks/humanoid_pathological_gait/assets.py`,
which searches in this order:

1. this package's `data/` directory (staged, the normal case),
2. `$SW_STROKEGAIT_DATA_DIR`, for site-specific layouts,
3. a `data/` directory beside the extension — useful when this repo is a submodule of the
   project that generates the data.

### `retargeted_h1_stride.npz` — required

One post-stroke gait cycle retargeted onto the H1's 19 joints by a Pinocchio QP solver with a
foot-clearance constraint. This is the reference the policy tracks, and the pose it resets
onto. Committed to this repository (88 KB).

| Key | Shape | Meaning |
| --- | --- | --- |
| `q_trajectory` | `(1000, 19)` | Joint positions, rad, in clinical order. |
| `v_trajectory` | `(1000, 19)` | Joint velocities, rad/s. Derived by finite difference if absent. |
| `time_vector` | `(1000,)` | Seconds; spans 1.2 s, one stride. |
| `joint_names` | `(19,)` | Clinical joint order, for provenance. |

The stride is from a **left-paretic** subject. Environments assigned a right-paretic side read
a sagittally mirrored copy, so one policy learns both presentations. The mirror negates the
yaw- and roll-axis joints and the torso, and leaves the pitch-axis joints alone — see
`h1_joints.py`.

### `stroke_gait_dataset.npz` — optional but recommended

407 parsed strides from 50 subjects, used as the AMP discriminator's expert motion corpus.
**Git-ignored at 31 MB** — it is clinical data and too large to commit.

| Key | Shape | Meaning |
| --- | --- | --- |
| `LHip`, `LKnee`, `LAnkle`, `RHip`, `RKnee`, `RAnkle`, `Pelvis` | `(407, 1000, 3)` | Joint angles, degrees, in Vicon Plug-in Gait convention. |
| `subject_ids`, `stride_indices`, `paretic_sides` | `(407,)` | Per-stride metadata. |
| `subject_characteristics` | `(50, 10)` | Per-subject clinical descriptors. |
| `char_keys` | `(10,)` | Names of those descriptors. |

Without it, the AMP buffer falls back to the reference stride, then to a synthetic prior.
Training runs either way, but the adversarial term is much weaker. `scripts/setup.sh` reports
which case you are in.

## Joint ordering

Two orderings are in play, and conflating them silently corrupts the task.

- **Clinical order** — how the data files are stored: left leg, right leg, torso, left arm,
  right arm.
- **Simulation order** — what the H1 articulation reports, which interleaves the two sides
  (`left_hip_yaw`, `right_hip_yaw`, `torso`, `left_hip_roll`, …).

`H1JointLayout` in `h1_joints.py` owns the conversion. Everything downstream of the reference
manager works in simulation order so it can be compared elementwise against
`robot.data.joint_pos`. The one deliberate exception is `get_amp_kinematic_tensors()`, which
returns joint tensors in **clinical** order: the expert corpus is stored clinically, and a
discriminator handed two different permutations would separate agent from expert on the
permutation alone, making the style reward meaningless.

## Regenerating

Both files come from an upstream clinical pipeline (originally the `sw-humanoid-strokegait`
project) that parses Vicon Plug-in Gait MATLAB exports and runs the QP retargeting. That
pipeline is not part of this extension. To use your own capture data, produce files matching
the schemas above and place them in this package's `data/` directory, or point
`SW_STROKEGAIT_DATA_DIR` at them.

## Ethics and licensing

The gait corpus is derived from human subject data. It is git-ignored here deliberately —
confirm your own ethics approval and data-sharing terms before distributing it, and note that
publishing this extension does not publish the corpus with it.
