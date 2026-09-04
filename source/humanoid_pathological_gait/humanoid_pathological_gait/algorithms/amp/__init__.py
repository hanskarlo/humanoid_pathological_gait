"""
Adversarial Motion Prior (AMP) algorithms and components.
"""

from .discriminator import (
    AMP_FEATURE_DIM,
    AMP_TRANSITION_DIM,
    AMPAgentReplayBuffer,
    AMPDiscriminator,
    AMPExpertMotionBuffer,
    AMPLossManager,
    extract_amp_features,
)

__all__ = [
    "AMP_FEATURE_DIM",
    "AMP_TRANSITION_DIM",
    "extract_amp_features",
    "AMPDiscriminator",
    "AMPLossManager",
    "AMPExpertMotionBuffer",
    "AMPAgentReplayBuffer",
]
