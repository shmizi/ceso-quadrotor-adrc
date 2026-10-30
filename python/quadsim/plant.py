import numpy as np

from .maths import UP, rotation
from .scenario import disturbance

# state layout (17 values):
#   0-2   position (m), world frame
#   3-5   velocity (m/s), world frame
#   6-9   orientation quaternion [w, x, y, z]
#   10-12 body angular rates [p, q, r] (rad/s)
#   13-16 actual thrust of the four motors (N)


def initial_state(c):
    # start at the configured position, level, with each motor holding a quarter of the weight
    state = np.zeros(17)
    state[:3], state[6] = c["initial_position"], 1.
    state[13:17] = c["mass"]*c["gravity"]/4
    return state


def actual_parameters(c):
    # mass and inertia of the real drone; with mismatch on, they differ from what the controller assumes
    mass, inertia = c["mass"], np.array(c["inertia"])

    if c["enable_mismatch"]:
        mass *= c["mass_mismatch_scale"]
        inertia *= c["inertia_mismatch_scale"]

    return mass, inertia


def plant_rhs(t, state, target, c, mixer):
    # time derivative of the state
    # target: motor thrusts the controller asked for (N)
    # mixer: B matrix that turns motor thrusts into [total thrust, roll, pitch, yaw torque]
    mass, inertia = actual_parameters(c)
    v, q, omega, thrusts = state[3:6], state[6:10], state[10:13], state[13:17]

    # wrench: [total thrust, three body torques]; force, torque: wind gusts
    wrench = mixer @ thrusts
    force, torque = disturbance(t, c["enable_disturbance"])

    acceleration = (rotation(q)[:, 2]*wrench[0]+force-c["linear_drag"]*v)/mass-c["gravity"]*UP
    angular = (wrench[1:]+torque-c["angular_drag"]*omega-np.cross(omega, inertia*omega))/inertia

    # quaternion derivative from the body rates
    w, x, y, z = q
    wx, wy, wz = omega
    qdot = 0.5*np.array([-x*wx-y*wy-z*wz, w*wx+y*wz-z*wy,
                         w*wy+z*wx-x*wz, w*wz+x*wy-y*wx])

    # each motor moves toward its target with a first-order lag
    return np.r_[v, acceleration, qdot, angular, (target-thrusts)/c["motor_time_constant"]]


def plant_step(t, state, target, c, mixer):
    # one fourth-order Runge-Kutta step; a, b, d, e are the four slope estimates
    dt = c["dt"]

    a = plant_rhs(t, state, target, c, mixer)
    b = plant_rhs(t+dt/2, state+dt*a/2, target, c, mixer)
    d = plant_rhs(t+dt/2, state+dt*b/2, target, c, mixer)
    e = plant_rhs(t+dt, state+dt*d, target, c, mixer)

    result = state+dt*(a+2*b+2*d+e)/6
    result[6:10] /= np.linalg.norm(result[6:10])
    return result


def in_envelope(state):
    # false once the drone has blown up numerically, hit the ground or flown more than 50 m away
    return np.all(np.isfinite(state)) and state[2] >= 0.1 and np.linalg.norm(state[:3]) <= 50
