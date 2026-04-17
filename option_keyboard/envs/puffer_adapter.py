import sys
import gym
import numpy as np
import torch

from pufferlib import pufferl


def _to_scalar_action(action):
    if isinstance(action, torch.Tensor):
        return int(action.item())
    return int(action)


class PufferOptionKeyboardAdapter:
    """Adapts a pufferlib vec env to Option Keyboard's single-env API.

    Exposes: reset(), step(), num_resources(), set_learning_options(),
    observation_space, action_space — matching the ForagingWorld contract.

    Two cumulants are defined:
        0 → "success" (1 when the agent reaches the goal, else 0)
        1 → "progress" (1 when Manhattan distance to goal decreased, else 0)
    """

    def __init__(self, vecenv):
        self.vecenv = vecenv
        self.env = vecenv.driver_env
        self._num_resources = 2
        self._learning_options = np.ones(self._num_resources, dtype=np.float32)
        self._learning_options_enabled = False
        self._prev_distance = None

        obs_shape = self.env.single_observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(np.prod(obs_shape)),),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(self.env.single_action_space.n)

    def num_resources(self):
        return self._num_resources

    def set_learning_options(self, w, flag):
        self._learning_options = np.array(w, dtype=np.float32)
        self._learning_options_enabled = bool(flag)

    def reset(self):
        observations, _ = self.env.reset(seed=np.random.randint(0, 2**31 - 1))
        obs = observations[0].astype(np.float32)
        self._prev_distance = self._goal_distance(obs)
        return obs.flatten()

    def step(self, action):
        scalar_action = _to_scalar_action(action)
        observations, rewards, terminals, truncations, _ = self.env.step(
            [scalar_action]
        )

        obs = observations[0].astype(np.float32)
        base_reward = float(rewards[0])
        done = bool(terminals[0] or truncations[0])

        success = 1 if base_reward > 0 else 0
        distance = self._goal_distance(obs)
        progress = 0
        if (
            self._prev_distance is not None
            and distance is not None
            and distance < self._prev_distance
        ):
            progress = 1
        self._prev_distance = distance

        cumulants = np.array([success, progress], dtype=np.float32)
        if self._learning_options_enabled:
            reward = float(np.dot(cumulants, self._learning_options))
        else:
            reward = base_reward + 0.1 * progress

        info = {
            "food type": (int(cumulants[0]), int(cumulants[1])),
            "rewards": cumulants.tolist(),
            "base_reward": base_reward,
            "success": float(success),
        }
        return obs.flatten(), reward, done, info

    def close(self):
        self.vecenv.close()

    @staticmethod
    def _goal_distance(obs):
        goal_positions = np.argwhere(obs[0] == 0.5)
        agent_positions = np.argwhere(obs[1] > 0.0)
        if len(goal_positions) == 0 or len(agent_positions) == 0:
            return None
        goal_y, goal_x = goal_positions[0]
        agent_y, agent_x = agent_positions[0]
        return abs(int(goal_y) - int(agent_y)) + abs(int(goal_x) - int(agent_x))


def load_puffer_env(env_name, seed=0, num_envs=1):
    """Load a pufferlib environment wrapped for Option Keyboard."""
    old_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        args = pufferl.load_config(env_name)
    finally:
        sys.argv = old_argv
    args["vec"]["backend"] = "Serial"
    args["vec"]["num_envs"] = 1
    args["vec"]["seed"] = seed
    args["env"]["num_envs"] = num_envs
    args["train"]["seed"] = seed
    vecenv = pufferl.load_env(env_name, args)
    return PufferOptionKeyboardAdapter(vecenv)
