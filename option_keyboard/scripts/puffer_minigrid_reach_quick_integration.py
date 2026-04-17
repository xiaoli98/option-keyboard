import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from itertools import product
from types import SimpleNamespace

import gym
import numpy as np
import torch

from pufferlib import pufferl
from option_keyboard.core.utils import create_log_files, set_global_seed
from option_keyboard.dqn.dqn import dqn
from option_keyboard.option_keyboard.agent import keyboard_player
from option_keyboard.option_keyboard.learn import learn_options


def _to_scalar_action(action):
    if isinstance(action, torch.Tensor):
        return int(action.item())
    return int(action)


class PufferMinigridReachOptionKeyboardAdapter:
    """Adapts a pufferlib vec env to Option Keyboard's single-env API."""

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
        observations, rewards, terminals, truncations, _ = self.env.step([scalar_action])

        obs = observations[0].astype(np.float32)
        base_reward = float(rewards[0])
        done = bool(terminals[0] or truncations[0])

        success = 1 if base_reward > 0 else 0
        distance = self._goal_distance(obs)
        progress = 0
        if self._prev_distance is not None and distance is not None and distance < self._prev_distance:
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
        }
        return obs.flatten(), reward, done, info

    def close(self):
        self.vecenv.close()

    @staticmethod
    def _goal_distance(obs):
        # Channel semantics from minigrid_reach diagnostics:
        # goal marker in channel 0 is 0.5, agent is in channel 1 (>0)
        goal_positions = np.argwhere(obs[0] == 0.5)
        agent_positions = np.argwhere(obs[1] > 0.0)
        if len(goal_positions) == 0 or len(agent_positions) == 0:
            return None
        goal_y, goal_x = goal_positions[0]
        agent_y, agent_x = agent_positions[0]
        return abs(int(goal_y) - int(agent_y)) + abs(int(goal_x) - int(agent_x))


def _load_env(seed):
    # pufferl.load_config internally parses CLI flags from sys.argv.
    # Keep only the script path so this runner's flags are not consumed there.
    old_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        args = pufferl.load_config("puffer_minigrid_reach")
    finally:
        sys.argv = old_argv
    args["vec"]["backend"] = "Serial"
    args["vec"]["num_envs"] = 1
    args["vec"]["seed"] = seed
    args["env"]["num_envs"] = 1
    args["train"]["seed"] = seed
    vecenv = pufferl.load_env("puffer_minigrid_reach", args)
    return PufferMinigridReachOptionKeyboardAdapter(vecenv), args


def _read_pickle_records(path):
    records = []
    with open(path, "rb") as handle:
        while True:
            try:
                records.append(pickle.load(handle))
            except EOFError:
                break
    return records


def map_assumptions(seed):
    env, args = _load_env(seed)
    obs = env.reset()
    checks = {
        "uses_pufferl_load_env": True,
        "env_name": "puffer_minigrid_reach",
        "observation_shape": list(obs.shape),
        "observation_dtype": str(obs.dtype),
        "action_space_n": int(env.action_space.n),
        "supports_num_resources": int(env.num_resources()) == 2,
        "supports_set_learning_options": True,
        "has_food_type_in_info": False,
        "info_food_type_len": None,
        "done_is_bool": None,
        "puffer_config_backend": args["vec"]["backend"],
    }
    env.set_learning_options(np.array([1, 1]), True)
    step_obs, step_reward, step_done, info = env.step(0)
    checks["has_food_type_in_info"] = "food type" in info
    checks["info_food_type_len"] = len(info["food type"]) if "food type" in info else None
    checks["done_is_bool"] = isinstance(step_done, bool)
    checks["step_observation_shape"] = list(step_obs.shape)
    checks["step_reward_type"] = type(step_reward).__name__
    env.close()
    return checks


def smoke_test(seed, rollout_steps):
    env, _ = _load_env(seed)
    obs = env.reset()
    total_reward = 0.0
    episodes = 0
    food_type_values = set()
    done_count = 0

    for _ in range(rollout_steps):
        action = np.random.randint(env.action_space.n)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        food_type_values.add(tuple(info["food type"]))
        if done:
            done_count += 1
            episodes += 1
            obs = env.reset()

    env.close()
    return {
        "rollout_steps": rollout_steps,
        "episodes_completed": episodes,
        "done_count": done_count,
        "mean_reward_per_step": total_reward / max(1, rollout_steps),
        "unique_food_type_values": [list(x) for x in sorted(food_type_values)],
        "final_obs_shape": list(obs.shape),
    }


def run_stage1(seed, log_root, steps, test_interval, n_test_runs, device):
    env, _ = _load_env(seed)
    set_global_seed(seed)
    run_name = "puffer_minigrid_reach_ok_stage1"
    args = SimpleNamespace(exp_name=run_name, log_dir=log_root)
    log_dir, log_files = create_log_files(args, env.num_resources())

    env.set_learning_options(np.array([1, 1]), True)
    value_fns = learn_options(
        env=env,
        d=env.num_resources(),
        eps1=0.2,
        eps2=0.1,
        alpha=1e-4,
        gamma=0.9,
        max_ep_steps=100,
        device=device,
        training_steps=steps,
        batch_size=10,
        pretrained_options="",
        test_interval=test_interval,
        n_test_runs=n_test_runs,
        log_files=log_files,
        log_dir=log_dir,
    )
    env.set_learning_options(np.array([1, 1]), False)
    env.close()

    finite_params = True
    for vf in value_fns:
        for param in vf.q_net.parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                finite_params = False
                break

    best_paths = [
        os.path.join(log_dir, "saved_models", "best", f"value_fn_{i + 1}.pt")
        for i in range(2)
    ]
    current_paths = [
        os.path.join(log_dir, "saved_models", f"value_fn_{i + 1}.pt")
        for i in range(2)
    ]

    return {
        "log_dir": log_dir,
        "log_files": log_files,
        "finite_model_params": finite_params,
        "best_checkpoint_exists": all(os.path.exists(p) for p in best_paths),
        "checkpoint_exists": all(os.path.exists(p) for p in current_paths),
        "steps": steps,
    }


def run_stage2(seed, stage1_log_dir, stage1_log_files, log_root, steps, test_interval, n_test_runs, device):
    env, _ = _load_env(seed)
    set_global_seed(seed)

    run_name = "puffer_minigrid_reach_ok_stage2"
    args = SimpleNamespace(exp_name=run_name, log_dir=log_root)
    log_dir, log_files = create_log_files(args, env.num_resources())

    # Recreate Q_E from Stage 1 best checkpoints.
    from option_keyboard.core.value_function import ValueFunction

    d = env.num_resources()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    Q_E = [
        ValueFunction(
            input_dim=state_dim + d,
            action_dim=action_dim + 1,
            n_options=d,
            hidden=[64, 128],
            batch_size=10,
            gamma=0.9,
            alpha=1e-4,
        )
        for _ in range(d)
    ]
    for i in range(d):
        ckpt_path = os.path.join(stage1_log_dir, "saved_models", "best", f"value_fn_{i + 1}.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(stage1_log_dir, "saved_models", f"value_fn_{i + 1}.pt")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        Q_E[i].q_net.load_state_dict(checkpoint["Q"])
        Q_E[i].q_net.to(device)

    W = [x for x in product([-1, 0, 1], repeat=2) if sum(x) >= 0]
    W.remove((0, 0))
    W = np.array(W)

    keyboard_player(
        env=env,
        W=W,
        Q=Q_E,
        alpha=1e-4,
        eps=0.1,
        gamma=0.99,
        training_steps=steps,
        batch_size=10,
        pretrained_agent="",
        max_ep_steps=300,
        device=device,
        test_interval=test_interval,
        n_test_runs=n_test_runs,
        log_file=log_files["agent"],
        log_dir=log_dir,
    )
    env.close()

    records = _read_pickle_records(log_files["agent"])
    mean_returns = [float(np.mean(r["returns"])) for r in records if "returns" in r]

    return {
        "log_dir": log_dir,
        "log_file": log_files["agent"],
        "stage1_log_dir": stage1_log_dir,
        "stage1_log_file_11": stage1_log_files["1,1"],
        "agent_checkpoint_exists": os.path.exists(os.path.join(log_dir, "saved_models", "agent.pt")),
        "best_agent_checkpoint_exists": os.path.exists(
            os.path.join(log_dir, "saved_models", "best", "agent.pt")
        ),
        "num_eval_points": len(mean_returns),
        "mean_return_first": mean_returns[0] if mean_returns else None,
        "mean_return_last": mean_returns[-1] if mean_returns else None,
        "mean_return_slope_nonnegative": (
            None if len(mean_returns) < 2 else (mean_returns[-1] - mean_returns[0]) >= 0
        ),
    }


def run_dqn_baseline(seed, log_root, steps, test_interval, n_test_runs, device):
    env, _ = _load_env(seed)
    set_global_seed(seed)

    run_name = "puffer_minigrid_reach_dqn_quick"
    args = SimpleNamespace(exp_name=run_name, log_dir=log_root)
    log_dir, log_files = create_log_files(args, 0)

    dqn(
        env=env,
        eps=0.1,
        gamma=0.99,
        alpha=1e-3,
        device=device,
        training_steps=steps,
        batch_size=10,
        pretrained_agent="",
        test_interval=test_interval,
        n_test_runs=n_test_runs,
        log_file=log_files["agent"],
        log_dir=log_dir,
    )
    env.close()

    records = _read_pickle_records(log_files["agent"])
    mean_returns = [float(np.mean(r["returns"])) for r in records if "returns" in r]
    return {
        "log_dir": log_dir,
        "log_file": log_files["agent"],
        "agent_checkpoint_exists": os.path.exists(os.path.join(log_dir, "saved_models", "agent.pt")),
        "best_agent_checkpoint_exists": os.path.exists(
            os.path.join(log_dir, "saved_models", "best", "agent.pt")
        ),
        "num_eval_points": len(mean_returns),
        "mean_return_first": mean_returns[0] if mean_returns else None,
        "mean_return_last": mean_returns[-1] if mean_returns else None,
    }


def main():
    parser = argparse.ArgumentParser("Quick integration for Option Keyboard + puffer_minigrid_reach")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rollout-steps", type=int, default=300)
    parser.add_argument("--stage1-steps", type=int, default=5000)
    parser.add_argument("--stage2-steps", type=int, default=5000)
    parser.add_argument("--dqn-steps", type=int, default=5000)
    parser.add_argument("--stage1-test-interval", type=int, default=1000)
    parser.add_argument("--stage2-test-interval", type=int, default=1000)
    parser.add_argument("--dqn-test-interval", type=int, default=1000)
    parser.add_argument("--n-test-runs", type=int, default=5)
    parser.add_argument(
        "--log-root",
        type=str,
        default="/workspaces/PufferTank_re/puffertank/pufferlib/baselines/option-keyboard/results",
    )
    args = parser.parse_args()

    os.makedirs(args.log_root, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "seed": args.seed,
        "device": str(device),
    }

    report["map_assumptions"] = map_assumptions(args.seed)
    report["smoke_test"] = smoke_test(args.seed, args.rollout_steps)

    stage1 = run_stage1(
        seed=args.seed,
        log_root=args.log_root,
        steps=args.stage1_steps,
        test_interval=args.stage1_test_interval,
        n_test_runs=args.n_test_runs,
        device=device,
    )
    report["stage1"] = stage1

    stage2 = run_stage2(
        seed=args.seed,
        stage1_log_dir=stage1["log_dir"],
        stage1_log_files=stage1["log_files"],
        log_root=args.log_root,
        steps=args.stage2_steps,
        test_interval=args.stage2_test_interval,
        n_test_runs=args.n_test_runs,
        device=device,
    )
    report["stage2"] = stage2

    dqn_report = run_dqn_baseline(
        seed=args.seed,
        log_root=args.log_root,
        steps=args.dqn_steps,
        test_interval=args.dqn_test_interval,
        n_test_runs=args.n_test_runs,
        device=device,
    )
    report["dqn"] = dqn_report

    stage2_last = report["stage2"]["mean_return_last"]
    dqn_last = report["dqn"]["mean_return_last"]
    if stage2_last is not None and dqn_last is not None:
        report["hierarchical_vs_dqn_last_return_delta"] = stage2_last - dqn_last
    else:
        report["hierarchical_vs_dqn_last_return_delta"] = None

    out_path = os.path.join(args.log_root, "puffer_minigrid_reach_quick_integration_report.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nSaved report: {out_path}")


if __name__ == "__main__":
    main()
