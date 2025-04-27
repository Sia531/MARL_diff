import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from mpe2 import simple_tag_v3
from pettingzoo import AECEnv
from rich import print
from rich.console import Console
from stable_baselines3.common.noise import NormalActionNoise

from td3bc_diffsion import TD3_BC

console = Console()


class MultiDiscreteToContinuousWrapper(gym.Wrapper):
    def __init__(self, env):
        super(MultiDiscreteToContinuousWrapper, self).__init__(env)
        self.action_sizes = env.action_space.nvec
        self.total_actions = sum(self.action_sizes)
        self.low = np.zeros(self.total_actions)
        self.high = np.ones(self.total_actions)
        self.action_space = spaces.Box(self.low, self.high, dtype=np.float32)

    def action(self, action):
        discrete_actions = []
        start_idx = 0
        for size in self.action_sizes:
            continuous_action = action[start_idx : start_idx + size]
            discrete_action = np.argmax(continuous_action)
            discrete_actions.append(discrete_action)
            start_idx += size
        return discrete_actions

    def step(self, action):
        discrete_action = self.action(action)
        return self.env.step(discrete_action)


class PettingZooToGymWrapper(gym.Env):
    def __init__(self, pettingzoo_env: AECEnv):
        super(PettingZooToGymWrapper, self).__init__()
        self.pettingzoo_env = pettingzoo_env
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(pettingzoo_env.action_spaces["agent_0"].n,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=pettingzoo_env.observation_spaces["agent_0"].shape,
            dtype=np.uint8,
        )

    def reset(self):
        self.pettingzoo_env.reset()
        return self._get_observations()

    def step(self, action):
        rewards = {}
        infos = {}
        for agent in self.pettingzoo_env.agent_iter():
            print(self.pettingzoo_env.action_space(agent))
            _, reward, done, info = self.pettingzoo_env.step(action)

            rewards[agent] = reward
            infos[agent] = info
            if done:
                self.pettingzoo_env.reset()
        return self._get_observations(), rewards, done, infos

    def _get_observations(self):
        return {
            agent: self.pettingzoo_env.observe(agent)
            for agent in self.pettingzoo_env.agents
        }

    def render(self, mode="human"):
        self.pettingzoo_env.render()

    def close(self):
        self.pettingzoo_env.close()


if __name__ == "__main__":
    env_dir = os.path.join("./results", "simple_tag_v3")
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    total_files = len([file for file in os.listdir(env_dir)])
    result_dir = os.path.join(env_dir, f"{total_files + 1}")
    os.makedirs(result_dir)
    env = simple_tag_v3.env(render_mode="human")
    eval_env = simple_tag_v3.env(render_mode="human")
    env = PettingZooToGymWrapper(env)
    eval_env = PettingZooToGymWrapper(eval_env)
    n_actions = 5
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
    )

    model = TD3_BC("MlpPolicy", env, bc_coef=0.5, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=500, eval_env=eval_env, batch_size=1024)
