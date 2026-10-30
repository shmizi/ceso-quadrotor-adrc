import numpy as np

from .maths import matrix_exp, wrap


class ObserverBank:
    # Linear ESO (levels = 1) or cascade ESO (levels = 2) for the three axes of one loop.
    # bandwidth: observer bandwidth w0 per axis (rad/s)
    # ratio: spacing between cascade stages, so stage bandwidths are w0/ratio and w0
    # dt: sample time (s)
    # x: observer states, one column per axis; each stage has 3 rows:
    #    [tracking-error estimate, error-rate estimate, disturbance estimate]

    def __init__(self, bandwidth, levels, ratio, dt):
        n = 3*levels
        self.x = np.zeros((n, 3))
        self.matrices = []

        for wo in bandwidth:
            # a, b: continuous-time observer matrices for this axis
            a, b = np.zeros((n, n)), np.zeros((n, 2))

            for i in range(levels):
                # w: this stage's bandwidth; gain: its observer gains [3w, 3w^2, w^3]
                w = wo * ratio**(i+1-levels)
                gain = np.array([3*w, 3*w*w, w**3])
                j = 3*i

                a[j:j+3, j:j+3] = [[-gain[0], 1, 0], [-gain[1], 0, 1], [-gain[2], 0, 0]]
                b[j+1, 1] = -1

                if i == 0:
                    # the first stage is driven by the measured error
                    b[:3, 0] = gain
                else:
                    # later stages track the previous stage and add its disturbance estimate
                    a[j:j+3, j-3] += gain
                    a[j+1, 2:j:3] += 1

            # exact discretisation: [ad bd] = top rows of e^([a b; 0 0] * dt)
            augmented = np.zeros((n+2, n+2))
            augmented[:n, :n], augmented[:n, n:] = a, b
            discrete = matrix_exp(augmented * dt)
            self.matrices.append((discrete[:n, :n], discrete[:n, n:]))

        self.initialized = False

    def update(self, error, known):
        # error: measured tracking error per axis
        # known: the acceleration the controller already knows it applied
        # returns the error estimate, the error-rate estimate and the total disturbance f*
        if not self.initialized:
            self.x[::3, :] = error
            self.initialized = True

        for j, (ad, bd) in enumerate(self.matrices):
            self.x[:, j] = ad @ self.x[:, j] + bd @ np.array([error[j], known[j]])

        return self.x[-3, :], self.x[-2, :], np.sum(self.x[2::3, :], axis=0)


def tanh_fn(e, alpha, delta):
    # the paper's tanh function: tanh(alpha * ln(1/delta) * e)
    return np.tanh(alpha*np.log(1/delta)*e)


class TanhADRC:
    # Nonlinear ADRC from Cai et al.: tracking differentiator, tanh-ESO and tanh feedback law.
    # k: slope of each tanh term at zero, alpha * ln(1/delta)
    # beta: ESO gains (b0, b1, b2); beta01, beta02: feedback gains set by bandwidth wc
    # speed: tracking-differentiator speed r; lam: its gains (l1, l2)
    # v: [v1, v2], the smoothed reference and its rate
    # z: [z1, z2, z3], estimates of the output, its rate and the total disturbance

    def __init__(self, c, prefix, angular):
        self.dt, self.angular, self.delta = c["dt"], angular, c["tanh_delta"]
        self.alpha = c["tanh_alpha"]
        self.k = k = [a*np.log(1/self.delta) for a in self.alpha]

        wo = np.array(c[prefix+"_observer_bandwidth"])
        self.beta = (3*wo, 3*wo**2/k[0], wo**3/k[0])
        self.set_bandwidth(np.array(c[prefix+"_bandwidth"]))

        self.speed, self.lam = c[prefix+"_td_speed"], (1/k[0], 2/k[0])
        self.feedforward = bool(c["tanh_acceleration_feedforward"])
        self.v = self.z = None

    def set_bandwidth(self, wc):
        # wc: control bandwidth (rad/s); the TD3 agent changes it during flight
        self.beta01, self.beta02 = wc**2/self.k[1], 2*wc/self.k[2]

    def diff(self, a, b):
        # angle differences are wrapped to [-pi, pi)
        return wrap(a-b) if self.angular else a-b

    def update(self, desired, measured, known):
        dt, d, (a0, a1, a2) = self.dt, self.delta, self.alpha

        if self.v is None:
            self.v = [np.array(desired, float), np.zeros(3)]
            self.z = [np.array(measured, float), np.zeros(3), np.zeros(3)]

        (v1, v2), (z1, z2, z3), r = self.v, self.z, self.speed

        # tracking differentiator; vdot is the smoothed reference acceleration
        vdot = r*r*(-self.lam[0]*tanh_fn(self.diff(v1, desired), a0, d)
                    - self.lam[1]*tanh_fn(v2/r, a0, d))
        v2 = v2+dt*vdot
        v1 = v1+dt*v2

        # tanh-ESO; e is the estimation error and fe its tanh
        e = self.diff(z1, measured)
        fe = tanh_fn(e, a0, d)
        z1, z2, z3 = (z1+dt*(z2-self.beta[0]*e), z2+dt*(z3-self.beta[1]*fe+known),
                      z3+dt*(-self.beta[2]*fe))

        if self.angular:
            v1, z1 = wrap(v1), wrap(z1)
        self.v, self.z = [v1, v2], [z1, z2, z3]

        # feedback law u0, then cancel the estimated disturbance z3
        u0 = self.beta01*tanh_fn(self.diff(v1, z1), a1, d)+self.beta02*tanh_fn(v2-z2, a2, d)
        if self.feedforward:
            u0 = u0+vdot

        return u0-z3, vdot-z3
