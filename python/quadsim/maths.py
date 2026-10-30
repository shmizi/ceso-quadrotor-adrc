import numpy as np

# world z-axis, pointing up
UP = np.array([0., 0., 1.])


def wrap(a):
    # a: angle or array of angles in rad, returned in the range [-pi, pi)
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def rotation(q):
    # q: orientation quaternion [w, x, y, z]
    # returns R, the 3x3 matrix that turns body-frame vectors into world-frame vectors
    w, x, y, z = q / np.linalg.norm(q)

    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def euler(q):
    # q: orientation quaternion [w, x, y, z]
    # returns [roll, pitch, yaw] in rad (ZYX order), read off the rotation matrix r
    r = rotation(q)

    return np.array([np.arctan2(r[2, 1], r[2, 2]),
                     np.arcsin(np.clip(-r[2, 0], -1, 1)),
                     np.arctan2(r[1, 0], r[0, 0])])


def rate_map(a):
    # a: Euler angles [roll phi, pitch theta, yaw]
    # returns E, so that Euler-angle rates = E @ body angular rates [p, q, r]
    # s, c: sine and cosine of roll; ct: cosine of pitch
    phi, theta = a[:2]
    s, c, ct = np.sin(phi), np.cos(phi), np.cos(theta)

    # E divides by cos(pitch), so stop before pitch gets near 90 degrees
    if abs(ct) < 0.15:
        raise RuntimeError("Euler controller left its moderate-tilt envelope.")

    return np.array([[1, s*np.tan(theta), c*np.tan(theta)],
                     [0, c, -s],
                     [0, s/ct, c/ct]])


def allocation(c):
    # c: config dictionary
    # a: arm length projected on x and y, so the motors sit at (a, a), (-a, a), (-a, -a), (a, -a)
    # returns the mixer B: [total thrust, roll torque, pitch torque, yaw torque] = B @ [f1, f2, f3, f4]
    a = c["arm_length"] / np.sqrt(2)

    return np.vstack([np.ones(4),
                      [a, a, -a, -a],
                      [-a, a, a, -a],
                      c["yaw_moment_per_thrust"] * np.array([1, -1, 1, -1])])


def matrix_exp(a):
    # a: square matrix; returns e^a
    # scale: how many times a is halved so its norm is at most 0.5
    # term: the current Taylor-series term b^k / k!
    scale = max(0, int(np.ceil(np.log2(max(np.linalg.norm(a, 1), 0.5)/0.5))))
    b = a / (2.0**scale)

    result = np.eye(len(a))
    term = result.copy()
    for k in range(1, 65):
        term = term @ b / k
        result += term
        if np.linalg.norm(term, 1) < 1e-17:
            break

    # e^a = (e^(a / 2^scale))^(2^scale), so square the result back up
    for _ in range(scale):
        result = result @ result

    return result
