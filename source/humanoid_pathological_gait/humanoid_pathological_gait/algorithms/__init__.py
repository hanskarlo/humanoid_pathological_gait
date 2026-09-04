# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Learning algorithms used by this extension's training loop.

Isaac Lab's bundled ``rsl_rl`` has no Adversarial Motion Prior support, so the
pathological-gait policy is trained by a PPO+AMP loop that lives here rather than in
the RL library. Both sub-packages are plain PyTorch and depend on nothing from Isaac
Lab, which keeps them testable on CPU without booting a simulator.
"""
