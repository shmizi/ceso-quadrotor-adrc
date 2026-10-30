import numpy as np


class Noise:
    def __init__(self, seed):
        self.state = float(seed)

    def sample(self):
        u = np.zeros(6)
        for k in range(6):
            self.state = (16807.0 * self.state) % 2147483647.0
            u[k] = self.state / 2147483647.0

        r = np.sqrt(-2*np.log(u[::2]))
        phase = 2*np.pi*u[1::2]

        n = np.empty(6)
        n[::2], n[1::2] = r*np.cos(phase), r*np.sin(phase)
        return n


def reference(t, name="figure8"):
    s = max(0., t-3.)
    ramp, w, radius = 3., 0.35, 1.2
    r = min(s/ramp, 1.)
    phase = w*ramp*(2.5*r**4-3*r**5+r**6) if s < ramp else w*(s-ramp/2)
    yaw = 0.35*np.sin(phase/2)

    if name == "hover":
        return np.array([0., 0., 1.]), 0.

    if name == "circle":
        return np.array([radius*(1-np.cos(phase)), radius*np.sin(phase), 1+0.25*(1-np.cos(phase))]), yaw

    if name == "figure8":
        return np.array([radius*np.sin(phase), 0.65*radius*np.sin(2*phase), 1+0.25*(1-np.cos(phase))]), yaw

    if name == "waypoints":
        points = np.array([[0, 0, 1], [1, 0, 1.6], [1, 1, 1.6], [-1, 1, 1.2], [0, 0, 1.]])
        p = points[0].copy()
        for j, start in enumerate([2., 8., 14., 20.]):
            r = np.clip((t-start)/4., 0., 1.)
            p += (10*r**3-15*r**4+6*r**5)*(points[j+1]-points[j])
        return p, yaw

    raise ValueError(f"Unknown trajectory: {name}")


def pulse(t, start, end, edge=0.5):
    s = np.clip((t-start)/edge, 0., 1.)
    f = np.clip((end-t)/edge, 0., 1.)
    return (3*s*s-2*s**3)*(3*f*f-2*f**3)


def disturbance(t, enabled=True):
    if not enabled:
        return np.zeros(3), np.zeros(3)

    a, b = pulse(t, 9, 13), pulse(t, 18, 23)
    force = a*np.array([3., -2., 1.2]) + b*np.array([-2., 2.5, 0.6*np.sin(1.3*t)])
    torque = a*np.array([0.04, -0.03, 0.02]) + b*np.array([0.03*np.sin(t), 0.03, -0.02])
    return force, torque
