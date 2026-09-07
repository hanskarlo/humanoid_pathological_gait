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
    reset_to_reference = EventTerm(
        func=mdp.reset_to_reference_pose,
        mode="reset",
        params={
            "randomize_paretic_side": True,
            "randomize_phase": True,
            "position_noise": 0.02,
            "velocity_noise": 0.05,
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.2, 0.2)},
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
    # std stays at 0.15. Narrowing it to 0.04 to charge harder for crouching was measured
    # over a matched 14720-env run and made everything worse, including pelvis height:
    # mean 1.028 -> 1.012 m, oscillation 44.5 -> 82.3 mm, speed 0.151 -> -0.039 m/s. Past
    # ~80 mm of error the Gaussian is flat, so the term stopped shaping height at all.
    # See research_log/2026-09-07. Fix the target first; revisit the width only after.
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
        self.events.asymmetric_joint_gains = None
        self.events.asymmetric_leg_mass = None
        self.events.reset_to_reference.params["randomize_phase"] = False
        self.events.reset_to_reference.params["start_phase"] = 0.0
        self.events.reset_base.params["pose_range"] = {"yaw": (0.0, 0.0)}
        self.events.reset_base.params["velocity_range"] = {}

        # Spasticity and paretic weakness stay at full strength -- they are the phenomenon
        # under study, not a training aid.
        self.curriculum.spasticity = None
        self.curriculum.push_magnitude = None
        self.initial_spasticity_scale = 1.0
