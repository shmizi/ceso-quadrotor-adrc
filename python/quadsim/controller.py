import json

import numpy as np

from .config import MODES, ROOT, td3_settings
from .maths import UP, allocation, euler, rate_map, rotation, wrap
from .observers import ObserverBank, TanhADRC
from .scenario import Noise, reference


class Actor:
    def __init__(self, layers):
        self.layers = [(np.array(layer["W"], float), np.array(layer["b"], float)) for layer in layers]

    def __call__(self, x):
        for j, (w, b) in enumerate(self.layers):
            x = w @ x+b
            x = np.tanh(x) if j == len(self.layers)-1 else np.maximum(x, 0.)
        return x


def load_policy(c):
    path = ROOT/td3_settings(c)["policy"]
    if not path.exists():
        raise FileNotFoundError(f"No TD3 policy at {path}. Train one with python/td3_train.py.")

    with open(path, encoding="utf-8") as stream:
        return Actor(json.load(stream)["layers"])


def attitude_reference(force, yaw):
    dx, dy, dz = force/np.linalg.norm(force)
    sy, cy = np.sin(yaw), np.cos(yaw)
    return np.array([np.arcsin(np.clip(dx*sy-dy*cy, -1, 1)), np.arctan2(dx*cy+dy*sy, dz), yaw])


class Controller:
    def __init__(self, c, mode="ceso", policy="file"):
        if mode not in MODES:
            raise ValueError("Controller must be pid, eso, ceso, tanh or td3.")

        self.c, self.mode = c, mode
        levels = 2 if mode == "ceso" else 1
        self.observers = [ObserverBank(c[name+"_observer_bandwidth"], levels, c["cascade_ratio"], c["dt"])
                          for name in ("position", "attitude")]
        self.tanh = (TanhADRC(c, "position", False), TanhADRC(c, "attitude", True))

        self.mixer = allocation(c)
        self.mixer_inverse = np.linalg.inv(self.mixer)

        self.last_error, self.derivative, self.integral = np.zeros((3, 2, 3))
        self.pid_initialized = [False, False]
        self.saturated = False
        self.noise = Noise(c["seed"])

        self.policy = load_policy(c) if mode == "td3" and policy == "file" else policy
        h = self.td3 = td3_settings(c)
        self.period = int(round(h["agent_period"]/c["dt"]))
        self.max_step = h["max_gain_rate"]*c["dt"]
        self.count, self.obs = 0, np.zeros(15 if h["observe_gains"] else 12)
        self.att_integral, self.att_rate_error = np.zeros(3), np.zeros(3)
        self.att_wc = np.array(c["attitude_bandwidth"], float)
        self.wc_target = self.att_wc.copy()

    def set_td3_action(self, action):
        low, high = self.td3["action_range"]
        self.wc_target = low+(np.clip(action, -1., 1.)+1)*(high-low)/2

        if self.max_step <= 0:
            self.att_wc = self.wc_target
            self.tanh[1].set_bandwidth(self.att_wc)

    def update_gains(self):
        if self.policy is not None and self.count % self.period == 0:
            self.set_td3_action(self.policy(self.obs))

        if self.max_step > 0:
            self.att_wc = self.att_wc+np.clip(self.wc_target-self.att_wc, -self.max_step, self.max_step)
            self.tanh[1].set_bandwidth(self.att_wc)

    def observe(self, att_error, angles):
        self.att_integral = np.clip(self.att_integral+self.c["dt"]*att_error, -0.2, 0.2)
        self.att_rate_error = self.tanh[1].v[1]-self.tanh[1].z[1]
        self.obs = np.r_[att_error/0.05, self.att_integral/0.05, self.att_rate_error/0.5, angles/0.5]

        if self.td3["observe_gains"]:
            low, high = self.td3["action_range"]
            self.obs = np.r_[self.obs, 2*(self.att_wc-low)/(high-low)-1]

    def pid(self, error, wc, index):
        c, dt = self.c, self.c["dt"]

        if not self.pid_initialized[index]:
            self.last_error[index] = error
            self.pid_initialized[index] = True

        difference = error-self.last_error[index]
        if index == 1:
            difference = wrap(difference)

        alpha = np.exp(-(12., 35.)[index]*dt)
        self.derivative[index] = alpha*self.derivative[index]+(1-alpha)*difference/dt

        if not self.saturated:
            limit = (2., 0.3)[index]
            self.integral[index] = np.clip(self.integral[index]+dt*error, -limit, limit)

        self.last_error[index] = error
        ki = c[("pid_position_integral_gain", "pid_attitude_integral_gain")[index]]
        return wc**2*error+2*wc*self.derivative[index]+ki*self.integral[index]

    def feedback(self, desired, measured, known, index):
        if self.mode in ("tanh", "td3"):
            return self.tanh[index].update(desired, measured, known)

        error = desired-measured if index == 0 else wrap(desired-measured)
        wc = np.array(self.c[("position", "attitude")[index]+"_bandwidth"])

        if self.mode == "pid":
            return self.pid(error, wc, index), np.zeros(3)

        z1, z2, fstar = self.observers[index].update(error, known)
        return wc**2*z1+2*wc*z2+fstar, fstar

    def measure(self, state):
        c, q = self.c, state[6:10]
        p, angles, rmat = state[:3], euler(q), rotation(q)

        n = self.noise.sample()
        if c["enable_noise"]:
            p, angles = p+c["position_noise_std"]*n[:3], angles+c["attitude_noise_std"]*n[3:]

        return p, angles, rmat

    def position_loop(self, desired_p, p, rmat, wrench):
        c, g = self.c, self.c["gravity"]
        acc, fp = self.feedback(desired_p, p, rmat[:, 2]*wrench[0]/c["mass"]-g*UP, 0)

        acc[2] = np.clip(acc[2], -c["max_vertical_acceleration"], c["max_vertical_acceleration"])
        horizontal_limit = min(c["max_horizontal_acceleration"],
                               (g+acc[2])*np.tan(np.deg2rad(c["max_tilt_deg"])))
        horizontal_norm = np.linalg.norm(acc[:2])
        if horizontal_norm > horizontal_limit:
            acc[:2] *= horizontal_limit/horizontal_norm

        return c["mass"]*(acc+g*UP), fp

    def attitude_loop(self, desired_att, angles, wrench):
        c = self.c
        emap = rate_map(angles)
        acc, fa = self.feedback(desired_att, angles, emap @ (wrench[1:]/np.array(c["inertia"])), 1)

        limit = np.array(c["max_euler_acceleration"])
        command_ratio = np.max(np.abs(acc)/limit)
        acc = np.clip(acc, -limit, limit)

        return np.array(c["inertia"]) * np.linalg.solve(emap, acc), command_ratio, fa

    def allocate(self, force, rmat, torque):
        c = self.c
        requested = self.mixer_inverse @ np.r_[max(0., np.dot(force, rmat[:, 2])), torque]
        target = np.clip(requested, 0., c["max_thrust_to_weight"]*c["mass"]*c["gravity"]/4)
        self.saturated = bool(np.any(abs(target-requested) > 1e-9))
        return target

    def step(self, t, state, trajectory="figure8"):
        if self.mode == "td3":
            self.update_gains()
        self.count += 1

        p, angles, rmat = self.measure(state)
        desired_p, desired_yaw = reference(t, trajectory)
        wrench = self.mixer @ state[13:17]

        force, fp = self.position_loop(desired_p, p, rmat, wrench)
        desired_att = attitude_reference(force, desired_yaw)
        torque, command_ratio, fa = self.attitude_loop(desired_att, angles, wrench)

        att_error = wrap(desired_att-angles)
        if self.mode == "td3":
            self.observe(att_error, angles)

        target = self.allocate(force, rmat, torque)
        return target, {"reference": desired_p, "attitude_reference": desired_att,
                        "estimate": np.r_[fp, fa], "saturated": self.saturated,
                        "attitude_error": att_error, "command_ratio": command_ratio,
                        "gains": self.att_wc.copy()}
