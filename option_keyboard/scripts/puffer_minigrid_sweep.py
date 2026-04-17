import argparse
import os
import pickle
from statistics import mean
from types import SimpleNamespace

import wandb
from option_keyboard.option_keyboard.main import run_training


def _read_pickle_records(path):
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "rb") as handle:
        while True:
            try:
                records.append(pickle.load(handle))
            except EOFError:
                break
    return records


def _latest_agent_metrics(exp_dir):
    records = _read_pickle_records(os.path.join(exp_dir, "agent_log_file"))
    if not records:
        return {}

    last = records[-1]
    returns = [float(x) for x in last.get("returns", [])]
    success = [float(x) for x in last.get("success", []) if x is not None]
    episode_return = [float(x) for x in last.get("episode_return", []) if x is not None]

    metrics = {
        "ok_agent_last_step": int(last.get("steps", 0)),
        "ok_agent_mean_return": mean(returns) if returns else 0.0,
        "ok_agent_mean_success": mean(success) if success else 0.0,
        "ok_agent_mean_episode_return": mean(episode_return) if episode_return else 0.0,
        "ok_agent_num_eval_returns": len(returns),
    }
    return metrics


def _build_training_args(args, run):
    cfg = run.config
    exp_name = f"puffer_wb_{run.id}"
    train_args = SimpleNamespace(
        env_name=args.env_name,
        seed=int(cfg.seed),
        exp_name=exp_name,
        n_test_runs=int(cfg.n_test_runs),
        log_dir=args.log_dir,
        gamma_ok=float(cfg.gamma_ok),
        eps1_ok=float(cfg.eps1_ok),
        eps2_ok=float(cfg.eps2_ok),
        alpha_ok=float(cfg.alpha_ok),
        max_steps_ok=int(cfg.max_steps_ok),
        n_training_steps_ok=int(cfg.n_training_steps_ok),
        ok_batch_size=int(cfg.ok_batch_size),
        pretrained_options="",
        test_interval_option=int(cfg.test_interval_option),
        n_training_steps_agent=int(cfg.n_training_steps_agent),
        agent_batch_size=int(cfg.agent_batch_size),
        eps_agent=float(cfg.eps_agent),
        gamma_agent=float(cfg.gamma_agent),
        alpha_agent=float(cfg.alpha_agent),
        max_steps_agent=int(cfg.max_steps_agent),
        test_interval_agent=int(cfg.test_interval_agent),
        pretrained_agent="",
        scenario=int(args.scenario),
    )
    return train_args


def _sweep_configuration(args):
    # Ranged values for automatic search.
    return {
        "method": args.sweep_method,
        "name": args.sweep_name,
        "metric": {
            "goal": "maximize",
            "name": "ok_agent_mean_return",
        },
        "parameters": {
            "seed": {"values": args.seeds},
            "n_training_steps_ok": {
                "values": [args.steps_ok_min, args.steps_ok_mid, args.steps_ok_max]
            },
            "n_training_steps_agent": {
                "values": [args.steps_agent_min, args.steps_agent_mid, args.steps_agent_max]
            },
            "test_interval_option": {"values": [args.test_interval_option]},
            "test_interval_agent": {"values": [args.test_interval_agent]},
            "max_steps_ok": {"values": [args.max_steps_ok]},
            "max_steps_agent": {"values": [args.max_steps_agent]},
            "n_test_runs": {"values": [args.n_test_runs]},
            "ok_batch_size": {"values": [args.ok_batch_size]},
            "agent_batch_size": {"values": [args.agent_batch_size]},
            "alpha_ok": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-3},
            "alpha_agent": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-3},
            "gamma_ok": {"distribution": "uniform", "min": 0.90, "max": 0.999},
            "gamma_agent": {"distribution": "uniform", "min": 0.97, "max": 0.999},
            "eps1_ok": {"distribution": "uniform", "min": 0.10, "max": 0.40},
            "eps2_ok": {"distribution": "uniform", "min": 0.05, "max": 0.25},
            "eps_agent": {"distribution": "uniform", "min": 0.05, "max": 0.30},
        },
    }


def _train_one_run(args):
    with wandb.init(project=args.project, entity=args.entity, sync_tensorboard=True) as run:
        train_args = _build_training_args(args, run)
        exp_name = train_args.exp_name
        run.log({"ok_launch": 1})
        return_code = 0
        output = {}
        try:
            output = run_training(train_args) or {}
        except Exception:
            return_code = 1

        exp_dir = output.get("log_dir", os.path.join(args.log_dir, exp_name))
        metrics = _latest_agent_metrics(exp_dir)
        if metrics:
            # Log target metric at top-level for W&B sweep optimization.
            run.log(metrics)
            run.summary.update(metrics)

        run.summary["ok_exp_name"] = exp_name
        run.summary["ok_exp_dir"] = exp_dir
        run.summary["ok_return_code"] = return_code

        if return_code != 0:
            run.log({"ok_agent_mean_return": -1e9, "ok_failed": 1})
            raise RuntimeError(f"Training failed for {exp_name} with return code {return_code}")


def main():
    parser = argparse.ArgumentParser(
        description="W&B automatic sweep for Option Keyboard on puffer_minigrid_reach."
    )
    parser.add_argument("--project", type=str, default="puffer_option_keyboard")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--env-name", type=str, default="puffer_minigrid_reach")
    parser.add_argument("--log-dir", type=str, default="./results")
    parser.add_argument("--sweep-method", type=str, default="random", choices=["random", "bayes", "grid"])
    parser.add_argument("--sweep-name", type=str, default="puffer_ok_auto_sweep")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--count", type=int, default=20, help="Number of W&B agent runs.")
    parser.add_argument("--sweep-id", type=str, default="", help="Use existing sweep id if provided.")
    parser.add_argument("--create-only", action="store_true", help="Create sweep and exit.")
    parser.add_argument("--print-config", action="store_true", help="Print sweep config and exit.")

    # Training budget anchors (helps because this env is long-horizon).
    parser.add_argument("--steps-ok-min", type=int, default=200_000)
    parser.add_argument("--steps-ok-mid", type=int, default=500_000)
    parser.add_argument("--steps-ok-max", type=int, default=2_000_000)
    parser.add_argument("--steps-agent-min", type=int, default=200_000)
    parser.add_argument("--steps-agent-mid", type=int, default=500_000)
    parser.add_argument("--steps-agent-max", type=int, default=2_000_000)
    parser.add_argument("--test-interval-option", type=int, default=5_000)
    parser.add_argument("--test-interval-agent", type=int, default=5_000)
    parser.add_argument("--max-steps-ok", type=int, default=100)
    parser.add_argument("--max-steps-agent", type=int, default=200)
    parser.add_argument("--n-test-runs", type=int, default=10)
    parser.add_argument("--ok-batch-size", type=int, default=10)
    parser.add_argument("--agent-batch-size", type=int, default=10)
    parser.add_argument("--scenario", type=int, default=1)
    args = parser.parse_args()

    args.env_name = args.env_name
    os.makedirs(args.log_dir, exist_ok=True)
    wandb.tensorboard.patch(root_logdir=args.log_dir)

    sweep_config = _sweep_configuration(args)
    if args.print_config:
        print(sweep_config)
        return

    sweep_id = args.sweep_id
    if not sweep_id:
        sweep_id = wandb.sweep(sweep=sweep_config, project=args.project, entity=args.entity)
        print(f"Created sweep: {sweep_id}")
    else:
        print(f"Using existing sweep: {sweep_id}")

    if args.create_only:
        return

    wandb.agent(
        sweep_id=sweep_id,
        function=lambda: _train_one_run(args),
        count=args.count,
        project=args.project,
        entity=args.entity,
    )


if __name__ == "__main__":
    main()
