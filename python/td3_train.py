import argparse
import csv
import json
import time

import numpy as np

from quadsim import ROOT, TRAJECTORIES, Controller, load_config, td3_settings
from quadsim.plant import in_envelope, initial_state, plant_step
from quadsim.td3 import Agent


def reward(c, diag, control, gain_change, thrust_change):
    h, deg = control.td3, np.rad2deg
    w = h["reward_weights"]

    return (-(w[0]*np.sum(np.abs(deg(diag["attitude_error"])))
              + w[1]*np.sum(np.abs(deg(control.att_integral)))
              + w[2]*np.sum(np.abs(deg(control.att_rate_error))))
            + w[3]*float(diag["command_ratio"] <= 1.)
            - h["gain_change_weight"]*np.sum(np.abs(gain_change))/c["dt"]
            - h["rotor_activity_weight"]*np.sum(np.abs(thrust_change))/c["dt"])


def episode(c, trajectory, choose, store=None, on_step=None):
    control = Controller(c, "td3", policy=None)
    state, dt, period = initial_state(c), c["dt"], control.period
    steps = int(round(c["duration"]/dt))
    total, accumulated, schedule, crashed = 0., 0., [], False
    obs = action = None

    for k in range(steps):
        t = k*dt

        if k % period == 0:
            if obs is not None and store is not None:
                store(obs, action, accumulated/period, control.obs.copy(), 0.)
                if on_step:
                    on_step()

            obs = control.obs.copy()
            action = choose(obs)
            control.set_td3_action(action)
            schedule.append(control.wc_target.copy())
            accumulated = 0.

        try:
            gains, thrusts = control.att_wc.copy(), state[13:17].copy()
            target, diag = control.step(t, state, trajectory)
            state = plant_step(t, state, target, c, control.mixer)
            r = reward(c, diag, control, control.att_wc-gains, state[13:17]-thrusts)
            if not in_envelope(state):
                raise RuntimeError("envelope")
        except RuntimeError:
            crashed = True
            r = -control.td3["crash_penalty"]*period

        accumulated += r
        total += r/period

        if crashed:
            if store is not None:
                store(obs, action, accumulated/period, control.obs.copy(), 1.)
            break

    return total, crashed, np.mean(schedule, axis=0)


def evaluate(base, h, agent):
    c = dict(base, duration=h["episode_duration"], enable_mismatch=False)
    results = [episode(c, name, agent.act) for name in h["evaluation_trajectories"]]

    return (sum(r[0] for r in results), any(r[1] for r in results),
            np.mean([r[2] for r in results], axis=0))


def save_policy(path, base, h, agent, extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"description": "TD3 actor for tanh-ADRC attitude NLSEF bandwidths (Cai et al. 2026, Sec. 4)",
            "variant": base["td3_variant"],
            "observation": "per channel [error/0.05 rad, integral/0.05 rad s, rate error/0.5 rad/s, angle/0.5 rad]"
                           + (", then applied wc scaled to [-1,1]" if h["observe_gains"] else ""),
            "action": "tanh output in [-1,1] mapped linearly to td3.action_range for roll, pitch, yaw",
            "td3": h, **extra, "layers": agent.actor.export()}
    path.write_text(json.dumps(data))


def train(episodes=None, variant=None):
    base = load_config()
    base["td3_variant"] = variant or base["td3_variant"]
    h = td3_settings(base)
    episodes = episodes or h["episodes"]

    rng = np.random.default_rng(h["training_seed"])
    agent = Agent(h, rng)

    policy_path, curve_path = ROOT/h["policy"], ROOT/h["training_log"]
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    best, start = -np.inf, time.time()

    with curve_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["episode", "trajectory", "mismatch", "return", "crashed",
                         "mean_wc_roll", "mean_wc_pitch", "mean_wc_yaw", "evaluation_return"])

        for n in range(1, episodes+1):
            c = dict(base, duration=h["episode_duration"], seed=int(rng.integers(1, 2**31-2)),
                     enable_mismatch=bool(rng.random() < 0.5))
            trajectory = TRAJECTORIES[rng.integers(len(TRAJECTORIES))]
            total, crashed, wc = episode(c, trajectory, agent.explore, agent.buffer.add, agent.train_step)

            evaluation = ""
            if n % h["evaluation_interval"] == 0 or n == episodes:
                evaluation, eval_crash, eval_wc = evaluate(base, h, agent)
                if not eval_crash and evaluation > best:
                    best = evaluation
                    save_policy(policy_path, base, h, agent, {"episode": n, "evaluation_return": evaluation})
                print(f"episode {n:4d}  return {total:9.1f}  eval {evaluation:9.1f}  "
                      f"eval wc {np.round(eval_wc, 2)}  best {best:9.1f}  {time.time()-start:6.0f} s", flush=True)

            writer.writerow([n, trajectory, c["enable_mismatch"], total, crashed, *wc, evaluation])
            stream.flush()

    print("Saved best actor to", policy_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the TD3 attitude-gain tuner.")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--variant", choices=["improved", "paper"], default=None)
    args = parser.parse_args()
    train(args.episodes, args.variant)
