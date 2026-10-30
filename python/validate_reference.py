import json

import numpy as np

from quadsim import MODES, ROOT, csv_rows, export_csv, load_config, simulate
from quadsim.maths import allocation, matrix_exp, rotation
from quadsim.observers import ObserverBank, tanh_fn


def unit_checks(c):
    mixer = allocation(c)
    hover = c["mass"]*c["gravity"]
    assert np.allclose(mixer @ np.full(4, hover/4), [hover, 0, 0, 0])

    wrench = np.array([25., .1, -.2, .05])
    assert np.allclose(mixer @ np.linalg.solve(mixer, wrench), wrench)

    r = rotation(np.array([.85, .2, -.1, .3]))
    assert np.allclose(r.T @ r, np.eye(3)) and np.linalg.det(r) > 0

    a = np.array([[0., 1.], [0., 0.]])
    assert np.allclose(matrix_exp(a), np.eye(2)+a, atol=1e-14)

    e, alpha, delta = np.linspace(-2, 2, 81), .6, .01
    paper = -(delta**(alpha*e)-delta**(-alpha*e))/(delta**(alpha*e)+delta**(-alpha*e))
    assert np.allclose(tanh_fn(e, alpha, delta), paper)

    clean = dict(c, enable_noise=False, enable_disturbance=False, duration=1.)
    for mode in MODES:
        log = simulate(clean, mode, "hover")
        assert np.max(abs(log["state"][:, :3]-[0, 0, 1])) < 1e-11

    dt, disturbance = c["dt"], np.array([1.2, -.8, .5])
    for levels in (1, 2):
        observer = ObserverBank([10, 10, 10], levels, 2., dt)
        y, velocity, u = np.zeros(3), np.zeros(3), np.zeros(3)

        for _ in range(4000):
            z1, z2, fstar = observer.update(-y, u)
            u = 2.25*z1+3*z2+fstar
            acc = u+disturbance
            y += dt*velocity+0.5*dt**2*acc
            velocity += dt*acc

        assert np.max(abs(fstar+disturbance)) < .01
        assert np.max(abs(y)) < .005

    print("PASS allocation, quaternion rotation, exponential, Eq. 16 tanh, hover, disturbance estimation")


def parity(c, mode):
    log = simulate(c, mode, "figure8")
    export_csv(log, ROOT/"results"/f"python_{mode}_figure8.csv")
    entry = {"python_reference_metrics": log["metrics"]}

    matlab_path = ROOT/"results"/f"matlab_{mode}_figure8.csv"
    if not matlab_path.exists():
        entry["matlab_python_parity"] = "Run validate_matlab first, then rerun this script."
        return entry

    rows = int(round(5/c["dt"]))+1 if mode == "td3" else None
    matlab = np.loadtxt(matlab_path, delimiter=",", skiprows=1)[:rows]
    python = csv_rows(log)[:rows]

    discrepancy = float(np.max(abs(matlab-python)))
    assert discrepancy < 1e-6, f"MATLAB/Python {mode} parity failed: {discrepancy}"

    entry.update(parity_horizon_s=float(python[-1, 0]),
                 matlab_python_max_all_column_difference=discrepancy,
                 matlab_python_max_position_difference_m=float(np.max(abs(matlab[:, 1:4]-python[:, 1:4]))))
    print(f"PASS noisy/disturbed {python[-1, 0]:g}-second MATLAB/Python {mode} parity:", discrepancy)
    return entry


if __name__ == "__main__":
    c = load_config()
    unit_checks(c)

    report = {mode: parity(c, mode) for mode in ("ceso", "tanh", "td3")}
    (ROOT/"results"/"python_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
