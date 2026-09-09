# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for 19-DoF H1 post-stroke gait imitation.

Structured after Isaac Lab's own ``locomotion/velocity/config/h1`` task -- same
physics-preset pattern, same contact-sensor preset pattern, same flat-terrain scene --
with the velocity-command MDP replaced by clinical reference tracking:

* the action term commands residuals around the reference stride and injects the
  paretic limb's TSRT reflex torque,
* rewards score weighted joint tracking plus a real XCoM/margin-of-stability term
  built from contact-sensor readings,
* events reset onto the reference pose and randomize the two limbs asymmetrically.

PhysX is the default physics backend: the task leans on contact forces and
feed-forward joint efforts, and PhysX is the better-tested path for both. A
``newton_mjwarp`` preset is kept available via ``physics=newton_mjwarp``.
"""

from __future__ import annotations

from isaaclab_ovphysx.sensors import ContactSensorCfg as OvPhysXContactSensorCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.visualizers import VisualizerCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sensors import ContactSensorCfg as PhysXContactSensorCfg

from isaaclab_tasks.utils import PresetCfg

from ... import mdp
from ...assets import reference_stride_path
from ...h1_joints import FOOT_BODY_NAMES
from ...tsrt import TSRTParams

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import H1_MINIMAL_CFG  # isort: skip


##
# Physics and sensor presets
##


@configclass
class H1PathologicalPhysicsCfg(PresetCfg):
    """Physics presets. PhysX is the default; Newton/MJWarp is available but unvalidated here."""

    default = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=65,
            nconmax=15,
            cone="pyramidal",
            impratio=1,
            integrator="implicitfast",
        ),
        num_substeps=1,
        debug_mode=False,
    )
    physx = default


@configclass
class H1PathologicalContactSensorCfg(PresetCfg):
    """Backend-specific contact sensors, needed for the support-polygon computation."""

    default = PhysXContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    newton_mjwarp = NewtonContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    physx = default
    ovphysx = OvPhysXContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)


##
# Scene definition
##


@configclass
class H1PathologicalSceneCfg(InteractiveSceneCfg):
    """Flat ground, the H1 humanoid, and whole-body contact sensing."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    def __post_init__(self) -> None:
        """Stiffen the leg chain enough to hold a commanded pose against gravity.

        Isaac Lab's H1 gains are tuned for *velocity tracking*, where a standing steady-state
        pose error costs nothing. This task is pose imitation, where it is the whole game, and
        the gains were never revisited. Measured by holding the reference's single-support
        pose with zero action (``scripts/single_leg_stance.py``): the pelvis sags 73 mm with
        5.45 deg of mean joint error, and only 28.5% of the hold is genuinely one-footed --
        the sag drops the swing foot back onto the ground. That is upstream of the crouch, the
        foot scuff and the contact pattern, and no reward term can reach it.

        Per-joint the error is not uniform, so neither is the fix:

        ============  ==================  ===========
        joint         steady-state error  old stiffness
        ============  ==================  ===========
        ankle         12.50 / 10.96 deg   **20**
        knee          8.53 / 7.93 deg     200
        hip_pitch     6.61 / 6.39 deg     200
        ============  ==================  ===========

        The ankle is an order of magnitude softer than the rest of the leg while carrying
        comparable load in single support, so it is scaled hardest: ankle 8x, leg joints 4x.
        Damping rises with the square root of stiffness so the damping ratio is preserved --
        raising stiffness alone would leave the joint less damped than it started.

        The response is sub-linear, and not because of torque limits: at 4x/2x the ankle sat
        at 26-32% of its effort ceiling and the knee at 6-8%, so nothing is saturating. The
        leg is a closed chain against the ground and raising one joint's stiffness
        redistributes the error rather than removing it. That is also why there is no point
        chasing this much further.

        The arms are left alone: they carry no ground reaction and track fine.
        """
        legs = self.robot.actuators["legs"]
        legs.stiffness = {name: 4.0 * value for name, value in legs.stiffness.items()}
        legs.damping = {name: 2.0 * value for name, value in legs.damping.items()}

        feet = self.robot.actuators["feet"]
        feet.stiffness = {name: 8.0 * value for name, value in feet.stiffness.items()}
        feet.damping = {name: 2.83 * value for name, value in feet.damping.items()}

    contact_forces = H1PathologicalContactSensorCfg()

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=750.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Bounded joint-position residuals around the clinical reference stride."""

    joint_pos = mdp.ReferenceResidualSpasticActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        enable_spasticity=True,
    )


@configclass
class ObservationsCfg:
    """109-dimensional clinical state space."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group (order preserved)."""

        # base state (9)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        # proprioception (38)
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-1.5, n_max=1.5))
        # clinical reference target (38)
        reference_joint_pos = ObsTerm(func=mdp.reference_joint_pos)
        reference_joint_vel = ObsTerm(func=mdp.reference_joint_vel)
        # gait conditioning (5)
        paretic_side = ObsTerm(func=mdp.paretic_side)
        gait_phase = ObsTerm(func=mdp.gait_phase)
        # Along-track and cross-track distance from where the reference's root should be at
        # this phase. The policy has no memory and never observes its own displacement, so
        # without this the root-tracking reward is not actable.
        root_progression_error = ObsTerm(func=mdp.root_progression_error)
        # action history (19)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    """Startup, reset and interval events."""

    # -- startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_torso_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (1 / 1.25, 1.25),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # -- reset
    # Order matters: the paretic side is drawn here, and the three randomization terms
    # below read it to decide which limb gets the tight ranges.
    # Places the floating base as well as the joints. There is deliberately no separate
    # ``reset_root_state_uniform`` term any more: it used to overwrite the base with the
    # articulation default and a zero-mean velocity, which discarded the reference's
    # +0.244 m/s of forward momentum and its 1.78-9.65 deg of pelvic obliquity on every
    # reset. See the docstring of ``mdp.reset_to_reference_pose``.
    reset_to_reference = EventTerm(
        func=mdp.reset_to_reference_pose,
        mode="reset",
        params={
            "randomize_paretic_side": True,
            "randomize_phase": True,
            "position_noise": 0.02,
            "velocity_noise": 0.05,
            # What the removed term used to provide, less the parts that fought the reference.
            "scatter_xy": 0.5,
            "randomize_yaw": True,
            "root_lin_vel_noise": 0.1,
            "root_ang_vel_noise": 0.1,
        },
    )

    asymmetric_joint_gains = EventTerm(func=mdp.randomize_asymmetric_joint_gains, mode="reset")
    asymmetric_effort_limits = EventTerm(func=mdp.randomize_asymmetric_effort_limits, mode="reset")
    asymmetric_leg_mass = EventTerm(func=mdp.randomize_asymmetric_leg_mass, mode="reset")

    # -- interval
    push_robot = EventTerm(
        func=mdp.push_biased_toward_paretic_side,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.4, 0.4)}, "paretic_bias": 1.45},
    )


@configclass
class RewardsCfg:
    """Imitation fidelity, dynamic balance, and whole-body regularization."""

    # -- clinical imitation
    joint_pos_tracking = RewTerm(func=mdp.joint_pos_tracking, weight=15.0, params={"std": 0.35})
    joint_vel_tracking = RewTerm(func=mdp.joint_vel_tracking, weight=2.0, params={"std": 2.0})
    # target_velocity=None tracks the reference stride's own speed (0.244 m/s for the
    # current stride) instead of a constant. The 0.5 that was here is twice that, and lost
    # to joint tracking at weight 15.0, so it only ever contributed a fixed shortfall.
    forward_velocity = RewTerm(
        func=mdp.track_forward_velocity, weight=2.0, params={"target_velocity": None, "std": 0.5}
    )
    # target_height=None tracks the reference pelvis per phase instead of a constant. The
    # pelvis rises and falls 30.0 mm over the stride, so a constant target asked the policy
    # to hold still vertically while every other term asked it to walk.
    #
    # std is back at 0.15 for the multi-seed baseline, which reproduces the `audited` run.
    #
    # 0.06 was tried and is the better-*designed* value -- the audit's discrimination check
    # scores a policy 39 mm low at 0.936 here against 0.660 there, so at 0.15 this term has a
    # correct target and almost no power to act on it. But tightening it moved the pelvis by
    # 1 mm (39 -> 38) while swinging knee symmetry -22.8% and temporal asymmetry -254%, which
    # is variance, not effect. The crouch is not reward-limited, so the width is not the
    # lever, and the two settings are probably statistically indistinguishable. Revisit once
    # the seed spread is known.
    #
    # The history below is still the reason not to reach for 0.04.
    # Narrowing to 0.04 against the old constant target made everything worse -- pelvis
    # 1.028 -> 1.012 m, oscillation 44.5 -> 82.3 mm, speed 0.151 -> -0.039 -- because a
    # constant target the reference itself misses by 15 mm twice a stride becomes unreachable
    # when sharpened, and the term stopped shaping height at all. Against a per-phase target
    # the reference scores 1.0 at any width, so narrowing now charges only for the crouch.
    #
    # 0.15 could not pay for that crouch: the audit scores a policy sitting 39 mm low at
    # 0.936 against the reference's 1.000, a gap of 0.064 -- the term had a correct target
    # and no discriminating power. At 0.06 the same crouch scores 0.660, a gap of 0.340.
    # Not 0.04: that costs gradient far from the target for little extra separation.
    base_height = RewTerm(func=mdp.track_base_height, weight=3.0, params={"target_height": None, "std": 0.15})

    # -- dynamic balance
    margin_of_stability = RewTerm(
        func=mdp.margin_of_stability,
        weight=1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=list(FOOT_BODY_NAMES), preserve_order=True),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(FOOT_BODY_NAMES), preserve_order=True),
            # Reference-measured margins per support state (MuJoCo FK, same convention as
            # the reward): +0.191 +/- 0.024 in double support, -0.082 +/- 0.029 in single.
            # The single 0.04 that was here was the reference's *mean*, a value it never
            # actually holds -- and paired with a one-sided formulation it made the shuffle
            # score better than the patient. See research_log/2026-09-07.
            "double_support_margin": 0.19,
            "single_support_margin": -0.08,
            "std": 0.05,
        },
    )
    # target_height is 0.131 m, the reference's own measured mean paretic ankle-link height
    # through swing -- not the 0.10 m that was here, at which the reference's own motion
    # scored 0.540 while a policy lifting 11 mm scored 0.807. Only airborne samples are
    # scored, so this constant matters on ~5% of steps; it is corrected because it was
    # measurably wrong, not because it is expected to move a metric on its own.
    #
    # Weight stays 1.0 and the gate stays on measured contact. Raising the weight to 3.0 and
    # gating on the reference schedule instead was tried and regressed everything (see
    # research_log/2026-09-07): it pays for raising the ankle link without unloading the
    # foot, so the policy toe-stood -- more planted, slower, and 3.4x the cost of transport.
    paretic_foot_clearance = RewTerm(
        func=mdp.paretic_foot_clearance,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=list(FOOT_BODY_NAMES), preserve_order=True),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(FOOT_BODY_NAMES), preserve_order=True),
            "target_height": 0.131,
            "std": 0.04,
        },
    )

    # Asks each foot to be off the ground when the reference says it should be. This used
    # to be the *only* term requiring a step at all, because paretic_foot_clearance paid
    # only once the foot was already airborne and so charged nothing for planting it; the
    # 2026-09-05 baseline duly shuffled at 0.82-0.89 double support against 0.37 in the
    # patients. That gate is now the reference schedule too, so the two terms push together
    # -- this one on *when* the foot leaves the ground, that one on *how high* it gets.
    # Weight 6.0, not the 3.0 of the first attempt. Two changes at once, deliberately: the
    # class rebalance took the contestable share of this term from 0.317 to 0.5, and the
    # weight doubles what that share is worth, so the incentive to step is ~3.0 reward units
    # against joint tracking's 15.0 rather than ~0.95. The first run failed because the term
    # could not be heard; testing the rebalance alone at 3.0 would risk the same null result
    # for the same reason.
    swing_timing = RewTerm(
        func=mdp.swing_timing,
        weight=6.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(FOOT_BODY_NAMES), preserve_order=True),
            "contact_threshold": 1.0,
            "paretic_weight": 2.0,
        },
    )

    # Ties position to phase. swing_timing says when each foot should be down; this says
    # where the robot should be by then, which is what fixes cadence and stride length.
    root_progression = RewTerm(func=mdp.track_root_progression, weight=6.0, params={"std": 0.15})

    # -- the strategy the actuator model cannot express
    #
    # Reduced paretic stance is the defining temporal signature of hemiparetic gait, and
    # every policy trained here has the opposite sign: the reference scores -15.87% on
    # stance_fraction_asymmetry_pct, this configuration +10.49% over three seeds. Modelling
    # the deficit peripherally -- 40% effort ceiling plus a stretch reflex -- makes the
    # paretic limb the *cheaper* limb to stand on, so the policy leaves it planted. Patients
    # will not commit weight to a limb they do not trust; that decision lives above the
    # actuator level and nothing in this environment represented it.
    #
    # Weight -3.0 is a judgement call, comparable to base_height and half of swing_timing,
    # which already asks the paretic foot to lift on schedule at 6.0 and is complied with
    # only 33% of the time. A null here does not rule out the mechanism at a larger weight;
    # it rules out this one. Recorded before the run so the number is not chosen afterwards.
    paretic_load_aversion = RewTerm(
        func=mdp.paretic_load_aversion,
        weight=-3.0,
        params={
            # FOOT_BODY_NAMES, not a regex: the reward flips columns for right-paretic
            # environments and that is only correct if column 0 is reliably the left foot.
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(FOOT_BODY_NAMES), preserve_order=True),
            "contact_threshold": 1.0,
        },
    )

    # -- shaping
    alive = RewTerm(func=mdp.is_alive, weight=2.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class TerminationsCfg:
    """Episode termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height_fall = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.65})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})
    torso_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0},
    )
    tracking_divergence = DoneTerm(func=mdp.reference_tracking_divergence, params={"max_rms_error": 1.0})


@configclass
class CurriculumCfg:
    """Ramps that fade the pathology in as training progresses."""

    spasticity = CurrTerm(
        func=mdp.spasticity_ramp,
        params={"start_step": 0, "num_steps": 24_000, "start_scale": 0.0, "end_scale": 1.0},
    )
    push_magnitude = CurrTerm(
        func=mdp.push_magnitude_ramp,
        params={"term_name": "push_robot", "start_step": 0, "num_steps": 24_000, "start_scale": 0.0, "end_scale": 1.0},
    )
    # DISABLED after measurement. Kept because the code and the negative result are both
    # worth reproducing; re-enable by restoring the CurrTerm below and setting
    # initial_assist_scale back to 1.0.
    #
    # It worked on the headline metric and for the wrong reason. Against a matched control
    # (same retuned gains, no assist) double support fell 0.8545 -> 0.8053, but the mean
    # single-support *episode* went 0.0943 -> 0.0840 s against the reference's 1.02 s: the
    # policy unloads more often and more briefly, not for longer. Margin of stability during
    # single support was unchanged at -0.231 against the patient's -0.082, so the balance
    # skill the assist exists to teach did not appear. Unaided swing_timing also came out
    # *worse* than the control (2.931 against 3.430) -- the signature of a policy that was
    # carried rather than taught, which this term's own docstring warned to watch for.
    #
    # balance_assist = CurrTerm(
    #     func=mdp.balance_assist_decay,
    #     params={"start_step": 0, "num_steps": 19_000, "start_scale": 1.0, "end_scale": 0.0},
    # )


##
# Environment configuration
##


@configclass
class H1PathologicalGaitEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the 19-DoF H1 post-stroke gait imitation environment."""

    sim: SimulationCfg = SimulationCfg(physics=H1PathologicalPhysicsCfg())
    scene: H1PathologicalSceneCfg = H1PathologicalSceneCfg(num_envs=4096, env_spacing=2.5)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # -- task-specific settings
    reference_stride_path: str = ""
    """Path to the retargeted reference stride; resolved in ``__post_init__``."""

    reference_stride_duration_s: float = 1.2
    """Fallback stride duration if the reference file carries no time vector."""

    gait_phase_rate_scale: float = 1.0
    """Multiplier on how fast the gait clock advances relative to real time."""

    initial_spasticity_scale: float = 1.0
    """Reflex gain before the curriculum takes over (the curriculum overwrites it on step 1)."""

    initial_assist_scale: float = 0.0
    """Balance-assist gain. 0.0 disables it; see the curriculum note above for why."""

    assist_stiffness: float = 400.0
    """Lateral restoring stiffness at the pelvis, N per metre of CoM offset from the feet."""

    assist_damping: float = 80.0
    """Lateral damping at the pelvis, N per m/s of CoM velocity."""

    assist_max_force: float = 150.0
    """Cap on the assist force, N. About a quarter of the robot's 55 kg weight -- enough to
    arrest a sideways fall, not enough to carry the robot or to substitute for a step."""

    tsrt_params: TSRTParams = TSRTParams()
    """Biomechanical parameters of the TSRT spasticity model."""

    def __post_init__(self) -> None:
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 20.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

        self.viewer.eye = (4.0, 4.0, 2.0)
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(4.0, 4.0, 2.0))

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        if not self.reference_stride_path:
            self.reference_stride_path = str(reference_stride_path())


@configclass
class H1PathologicalGaitEnvCfg_PLAY(H1PathologicalGaitEnvCfg):
    """Deterministic, small-scene variant for playback and evaluation."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        # Deterministic conditions: no observation noise, no pushes, fixed start phase.
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        # Evaluation is always unaided, whatever the curriculum was doing during training.
        self.initial_assist_scale = 0.0
        self.curriculum.balance_assist = None
        self.events.asymmetric_joint_gains = None
        self.events.asymmetric_leg_mass = None
        self.events.reset_to_reference.params["randomize_phase"] = False
        self.events.reset_to_reference.params["start_phase"] = 0.0
        # Fixed heading and no scatter, but the reference's own root velocity is kept: it is
        # part of the state being evaluated, not a perturbation. Zeroing it is what made the
        # ``--zero_actions`` reference playback score 0% survival.
        self.events.reset_to_reference.params["scatter_xy"] = 0.0
        self.events.reset_to_reference.params["randomize_yaw"] = False
        self.events.reset_to_reference.params["root_lin_vel_noise"] = 0.0
        self.events.reset_to_reference.params["root_ang_vel_noise"] = 0.0

        # Spasticity and paretic weakness stay at full strength -- they are the phenomenon
        # under study, not a training aid.
        self.curriculum.spasticity = None
        self.curriculum.push_magnitude = None
        self.initial_spasticity_scale = 1.0
