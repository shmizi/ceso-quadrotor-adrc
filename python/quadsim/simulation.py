import copy
import csv
from pathlib import Path

import numpy as np

from .config import load_config
from .controller import Controller
from .maths import euler, wrap
from .plant import in_envelope, initial_state, plant_step
from .scenario import disturbance

LOG_WIDTHS = {"state": 17, "euler": 3, "reference": 3, "attitude_reference": 3,
              "estimate": 6, "target": 4, "force": 3, "gains": 3}

CSV_HEADER = ["time_s", "x_m", "y_m", "z_m", "xref_m", "yref_m", "zref_m",
              "roll_rad", "pitch_rad", "yaw_rad", "rollref_rad", "pitchref_rad", "yawref_rad",
              "motor1_N", "motor2_N", "motor3_N", "motor4_N",
              "fstar_x", "fstar_y", "fstar_z", "fstar_roll", "fstar_pitch", "fstar_yaw", "saturated"]


def new_log(n, dt):
    log = {key: np.zeros((n, width)) for key, width in LOG_WIDTHS.items()}
    log.update(time=np.arange(n)*dt, saturated=np.zeros(n, dtype=bool))
    return log


def record(log, k, t, state, target, diag, c):
    log["state"][k], log["euler"][k], log["target"][k] = state, euler(state[6:10]), target
    for key in ("reference", "attitude_reference", "estimate", "saturated", "gains"):
        log[key][k] = diag[key]
    log["force"][k] = disturbance(t, c["enable_disturbance"])[0]


def metrics(log):
    error = log["reference"]-log["state"][:, :3]
    ae = wrap(log["attitude_reference"]-log["euler"])
    dt = log["time"][1]-log["time"][0]

    return {"position_rmse_m": float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
            "peak_position_error_m": float(np.max(np.linalg.norm(error, axis=1))),
            "attitude_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(np.sum(ae**2, axis=1))))),
            "rotor_thrust_variation_N": float(np.sum(np.abs(np.diff(log["state"][:, 13:17], axis=0)))),
            "rotor_effort_N2s": float(dt*np.sum(log["state"][:, 13:17]**2)),
            "saturated_sample_percent": float(100*np.mean(log["saturated"]))}


def simulate(config=None, mode="ceso", trajectory="figure8", duration=None):
    c = copy.deepcopy(config or load_config())
    if duration is not None:
        c["duration"] = duration

    n = int(round(c["duration"]/c["dt"]))+1
    if n < 2:
        raise ValueError("Duration must be at least one timestep.")

    control, state = Controller(c, mode), initial_state(c)
    log = new_log(n, c["dt"])

    for k, t in enumerate(log["time"]):
        target, diag = control.step(t, state, trajectory)
        record(log, k, t, state, target, diag, c)

        if k < n-1:
            state = plant_step(t, state, target, c, control.mixer)
            if not in_envelope(state):
                raise RuntimeError(f"Flight envelope exceeded at {t:.3f} s ({mode}, {trajectory}).")

    log["metrics"], log["config"], log["mode"], log["trajectory"] = metrics(log), c, mode, trajectory
    return log


def csv_rows(log):
    return np.column_stack([log["time"], log["state"][:, :3], log["reference"], log["euler"],
                            log["attitude_reference"], log["state"][:, 13:17], log["estimate"], log["saturated"]])


def export_csv(log, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(CSV_HEADER)
        writer.writerows(csv_rows(log))
