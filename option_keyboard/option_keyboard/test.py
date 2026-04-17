from option_keyboard.option_keyboard.ok import option_keyboard
import torch
import pickle
import numpy as np


def _to_scalar(value):
    try:
        if isinstance(value, (float, int, bool, np.floating, np.integer, np.bool_)):
            return float(value)
        arr = np.asarray(value)
        if arr.size == 0:
            return None
        return float(arr.reshape(-1)[-1])
    except Exception:
        return None


def _extract_metric_from_env_info(env_info_seq, key, ):
    def flatten_dict(prefix, value, out):
        if isinstance(value, dict):
            for k, v in value.items():
                next_prefix = f'{prefix}/{k}' if prefix else str(k)
                flatten_dict(next_prefix, v, out)
        else:
            out[prefix] = value

    values = []
    target_tail = key.lower()
    for step_info in env_info_seq:
        if not isinstance(step_info, dict):
            continue
        flat = {}
        flatten_dict('', step_info, flat)
        for k, v in flat.items():
            key_name = k.lower().split('/')[-1]
            if key_name != target_tail:
                continue
            scalar = _to_scalar(v)
            if scalar is not None:
                values.append(scalar)
    return np.mean(values) if values else None


def test_agent(env, W, Q_w, Q_E, gamma, training_steps, max_ep_steps, device,
               n_test_runs, test_log_file):

    """
        This function tests the overall performance of the agent, given that
        it has already learnt the keyboard.

        env: Environment
        W: Weight vector to be learnt over cumulants
        Q_w: Q-function over weight vectors
        Q_E: Q-functions over all cumulants and options
        gamma: Discount factor
        training_steps: Number of steps for which agent is to be trained
        max_ep_steps: Maximum number of steps in an episode
        device: cpu or gpu
        n_test_runs: Number of episodes for which performance is tested
        test_log_file: Path to log file for test results
    """
    ep_returns = []
    successes = []
    true_episode_returns = []

    for _ in range(n_test_runs):
        s = env.reset()
        s = torch.from_numpy(s).float().to(device)
        n_steps = 0
        ep_return = 0
        episode_env_info = []
        done = False

        while n_steps < max_ep_steps and not done:
            with torch.no_grad():
                q = Q_w(s)
            w = W[torch.argmax(q)]
            (s_next, done, r_prime,
             gamma_prime, n_steps, info) = option_keyboard(env, s, w, Q_E,
                                                           gamma, n_steps,
                                                           max_ep_steps,
                                                           device)

            s_next = torch.from_numpy(s_next).float().to(device)
            s = (s_next if not done
                 else torch.from_numpy(env.reset()).float().to(device))

            ep_return += sum(info['rewards'])
            episode_env_info.extend(info['env_info'])

        ep_returns.append(ep_return)
        successes.append(_extract_metric_from_env_info(
            episode_env_info, 'success'))
        info_ep_return = _extract_metric_from_env_info(
            episode_env_info, 'episode_return')
        true_episode_returns.append(info_ep_return if info_ep_return is not None
                                    else ep_return)

    # print('Steps:', training_steps,
    #       'Avg. return:', sum(ep_returns) / n_test_runs,
    #       'Episodic return:', ep_returns,
    #       'Success:', successes,
    #       'Info episode_return:', info_episode_returns)

    logfile = open(test_log_file, 'a+b')
    pickle.dump({'steps': training_steps,
                 'returns': ep_returns,
                 'success': successes,
                 'episode_return': true_episode_returns}, logfile)
    logfile.close()

    return ep_returns, successes, true_episode_returns


def test_learning_options(env, Q_E, index, w, gamma, training_steps,
                          max_ep_steps, device, n_test_runs,
                          log_file):

    """
        This function tests the performance of the agent for different weight
        vectors w. This is typically used to see how the options being learnt
        perform for w = (1, 1), which would optimize for all types of food
        items. We also record performance for w = (1, -1) and w = (-1, 1) since
        the keyboard learnt should perform reasonably well for all
        configurations of w that we consider.

        env: Environment
        Q_E: Q-functions over all cumulants and options
        index: Index of cumulant for which performance is measured
        w: Weight vector (kept constant to measure keyboard performance)
        gamma: Discount factor
        training_steps: Number of steps for which agent is to be trained
        max_ep_steps: Maximum number of steps in an episode
        device: cpu or gpu
        n_test_runs: Number of episodes for which performance is tested
        log_file: Path to log file for test results
    """
    ep_returns = []
    cumulant_returns = []
    successes = []
    true_episode_returns = []

    env.set_learning_options(w, True)

    for _ in range(n_test_runs):
        s = env.reset()
        s = torch.from_numpy(s).float().to(device)
        n_steps = 0
        ep_return = 0
        cumulant_return = 0
        episode_env_info = []
        done = False

        while n_steps < max_ep_steps and not done:
            (s_next, done, r_prime,
             gamma_prime, n_steps, info) = option_keyboard(env, s, w, Q_E,
                                                           gamma, n_steps,
                                                           max_ep_steps,
                                                           device)

            s_next = torch.from_numpy(s_next).float().to(device)
            s = (s_next if not done
                 else torch.from_numpy(env.reset()).float().to(device))

            ep_return += sum(info['rewards'])
            episode_env_info.extend(info['env_info'])

            cumulant_return += sum([info['env_info'][i]['rewards'][index]
                                    for i in range(len(info['env_info']))])

        ep_returns.append(ep_return)
        cumulant_returns.append(cumulant_return)
        successes.append(_extract_metric_from_env_info(
            episode_env_info, 'success'))
        info_ep_return = _extract_metric_from_env_info(
            episode_env_info, 'episode_return')
        true_episode_returns.append(info_ep_return if info_ep_return is not None
                                    else ep_return)

    print('w:', w, 'Steps:', training_steps,
          'Avg. return:', sum(ep_returns) / n_test_runs,
          'Episodic return:', ep_returns,
          'Cumulant return:', cumulant_returns,
          'Success:', successes,
          'True episode_return:', true_episode_returns)

    logfile = open(log_file, 'a+b')
    pickle.dump({'steps': training_steps, 'returns': ep_returns,
                 'cumulant_returns': cumulant_returns,
                 'success': successes,
                 'episode_return': true_episode_returns}, logfile)
    logfile.close()

    env.set_learning_options(np.ones(len(w)), True)

    return ep_returns, cumulant_returns, successes, true_episode_returns
