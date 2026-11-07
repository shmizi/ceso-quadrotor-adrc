import argparse
import csv
from multiprocessing import Pool

import numpy as np

from quadsim import ROOT, TRAJECTORIES, Controller, load_config
from quadsim.observers import ObserverBank
from quadsim.plant import in_envelope, initial_state, plant_step
from quadsim.simulation import metrics, new_log, record

METRICS = ["position_rmse_m", "peak_position_error_m", "attitude_rmse_deg", "rotor_thrust_variation_N"]
GUST_WINDOWS = ((8.5, 13.5), (17.5, 23.5))


def scale(c, name, ratio):
    return {name: [v*ratio for v in c[name]]}


def sweeps(c):
    # one setting at a time around config.json; values are the roll/pitch or x/y bandwidth,
    # and the other axes of that loop scale with it
    cases = []

    def add(study, name, values, nominal):
        cases.extend((study, value, scale(c, name, value/nominal), None) for value in values)

    add("attitude_wc", "attitude_bandwidth", (1, 3, 5, 7, 10, 14, 20, 28), c["attitude_bandwidth"][0])
    add("attitude_w0", "attitude_observer_bandwidth", (12, 18, 24, 30, 36, 48, 60, 80, 100),
        c["attitude_observer_bandwidth"][0])
    add("position_w0", "position_observer_bandwidth", (4, 6, 8, 10, 14, 20, 28),
        c["position_observer_bandwidth"][0])
    add("position_wc", "position_bandwidth", (0.5, 0.75, 1, 1.5, 2, 3, 4), c["position_bandwidth"][0])
    cases.extend(("cascade_ratio", value, {"cascade_ratio": value}, None) for value in (1.25, 1.5, 2, 3, 4))
    return cases


def error_sources():
    # what the attitude error is made of: remove one input at a time
    return [("source", name, overrides, None) for name, overrides in (
        ("all", {}),
        ("no position noise", {"position_noise_std": 0.}),
        ("no attitude noise", {"attitude_noise_std": 0.}),
        ("no noise", {"enable_noise": False}),
        ("no gusts", {"enable_disturbance": False}),
        ("no noise, no gusts", {"enable_noise": False, "enable_disturbance": False}))]


def schedules(c):
    # fixed position observer bandwidths against a schedule told the gust times in advance;
    # a schedule is ((attitude, position) scale in calm air, the same during a gust)
    fixed = [("schedule", f"fixed position {ratio}", scale(c, "position_observer_bandwidth", ratio), None)
             for ratio in (0.79, 0.82, 0.85, 0.88, 0.91, 0.94, 1.)]
    return fixed+[("schedule", "gust position 0.79-1", {}, ((1., .79), (1., 1.))),
                  ("schedule", "gust position 0.79-1, attitude 1-1.26", {}, ((1., .79), (1.26, 1.)))]


class GustScheduled(Controller):
    def __init__(self, c, calm, gust):
        super().__init__(c, "ceso")
        self.tables = [self.matrices(c, calm), self.matrices(c, gust)]

    @staticmethod
    def matrices(c, scales):
        return [ObserverBank(np.array(c[name+"_observer_bandwidth"])*ratio, 2, c["cascade_ratio"], c["dt"]).matrices
                for name, ratio in zip(("position", "attitude"), scales[::-1])]

    def step(self, t, state, trajectory="figure8"):
        table = self.tables[any(a <= t <= b for a, b in GUST_WINDOWS)]
        self.observers[0].matrices, self.observers[1].matrices = table
        return super().step(t, state, trajectory)


def fly(job):
    # returns the study, the setting, the noise seed and the flight's metrics (None after a crash)
    study, value, overrides, schedule, trajectory, mismatch, seed = job
    c = load_config()
    c.update(overrides, seed=seed, enable_mismatch=mismatch)

    control = Controller(c, "ceso") if schedule is None else GustScheduled(c, *schedule)
    state = initial_state(c)
    n = int(round(c["duration"]/c["dt"]))+1
    log = new_log(n, c["dt"])

    for k, t in enumerate(log["time"]):
        target, diag = control.step(t, state, trajectory)
        record(log, k, t, state, target, diag, c)
        if k < n-1:
            state = plant_step(t, state, target, c, control.mixer)
            if not in_envelope(state):
                return study, value, seed, None

    m = metrics(log)
    return study, value, seed, [m[key] for key in METRICS]


def summarise(results):
    # mean over paths and mismatch settings per noise seed, then mean and 95 % interval across seeds
    rows = []
    for study, value in dict.fromkeys((r[0], r[1]) for r in results):
        group = [r for r in results if r[0] == study and r[1] == value]
        good = [r for r in group if r[3] is not None]
        row = {"study": study, "setting": value, "flights": len(group), "crashes": len(group)-len(good)}

        seeds = sorted({r[2] for r in good})
        per_seed = np.array([np.mean([r[3] for r in good if r[2] == s], axis=0) for s in seeds])
        for j, key in enumerate(METRICS):
            row[key] = per_seed[:, j].mean()
            row[key+"_ci95"] = 1.96*per_seed[:, j].std(ddof=1)/np.sqrt(len(seeds))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="CESO sensitivity to its gains, its error sources and gust scheduling.")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--processes", type=int, default=14)
    args = parser.parse_args()

    c = load_config()
    seeds = [int(s) for s in np.random.default_rng(2027).integers(1, 2**31-2, args.seeds)]
    cases = sweeps(c)+error_sources()+schedules(c)
    jobs = [(*case, name, mismatch, seed) for case in cases
            for name in TRAJECTORIES for mismatch in (False, True) for seed in seeds]

    with Pool(args.processes) as pool:
        results = pool.map(fly, jobs, chunksize=1)

    rows = summarise(results)
    path = ROOT/"results"/"sensitivity.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(f"{row['study']:>14} {str(row['setting']):>38} {row['crashes']:>3}/{row['flights']:<3}  "
              + "  ".join(f"{row[key]:9.4f} ±{row[key+'_ci95']:.4f}" for key in METRICS))
    print("Saved", path)


if __name__ == "__main__":
    main()
