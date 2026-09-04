# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-H1-Pathological-Gait-v0",
    entry_point=f"{__name__.rsplit('.', 2)[0]}.h1_pathological_env:H1PathologicalGaitEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_pathological_env_cfg:H1PathologicalGaitEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-H1-Pathological-Gait-Play-v0",
    entry_point=f"{__name__.rsplit('.', 2)[0]}.h1_pathological_env:H1PathologicalGaitEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_pathological_env_cfg:H1PathologicalGaitEnvCfg_PLAY",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
