import numpy as np
import scipy.signal as signal

from utils import reshape_pt1, reshape_dim1, reshape_dim1_tonormal


# Possible controllers

# Sinusoidal control law, imposing initial value
def sin_controller_02D(t, kwargs, t0, init_control):
    gamma = kwargs.get('gamma')
    omega = kwargs.get('omega')
    if np.isscalar(t):
        if t == t0:
            u = reshape_pt1(init_control)
        else:
            u = reshape_pt1([[0, gamma * np.cos(omega * t)]])
    else:
        u = reshape_pt1(np.concatenate((reshape_dim1(
            np.zeros(len(t))), reshape_dim1(gamma * np.cos(omega * t))),
            axis=1))
        if t[0] == t0:
            u[0] = reshape_pt1(init_control)
    return u


# Sinusoidal control law, imposing initial value
def sin_controller_1D(t, kwargs, t0, init_control):
    gamma = kwargs.get('gamma')
    omega = kwargs.get('omega')
    if np.isscalar(t):
        if t == t0:
            u = reshape_pt1(init_control)
        else:
            u = reshape_pt1([[gamma * np.cos(omega * t)]])
    else:
        u = reshape_dim1(gamma * np.cos(omega * t))
        if t[0] == t0:
            u[0] = reshape_pt1(init_control)
    return u


# Chirp control law, imposing initial value
def chirp_controller(t, kwargs, t0, init_control):
    gamma = kwargs.get('gamma')
    f0 = kwargs.get('f0')
    f1 = kwargs.get('f1')
    t1 = kwargs.get('t1')
    nb_cycles = int(np.floor(np.min(t) / t1))
    t = t - nb_cycles * t1
    if np.isscalar(t):
        if t == t0:
            u = reshape_pt1(init_control)
        else:
            u = reshape_pt1(
                [[0, signal.chirp(t, f0=f0, f1=f1, t1=t1, method='linear')]])
    else:
        u = reshape_pt1(np.concatenate((reshape_dim1(np.zeros(len(t))),
                                        reshape_dim1(
                                            signal.chirp(t, f0=f0, f1=f1, t1=t1,
                                                         method='linear'))),
                                       axis=1))
        if t[0] == t0:
            u[0] = reshape_pt1(init_control)
    return gamma * u


# Fake control law just returning zeros (for when one needs to be defined for
# simulation but actually autonomous system)
def null_controller(t, kwargs, t0, init_control):
    if np.isscalar(t):
        t = np.array([t])
    else:
        t = reshape_dim1_tonormal(t)
    u = reshape_pt1(np.zeros((len(t), init_control.shape[1])))
    return u
