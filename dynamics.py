import numpy as np
from scipy.integrate import solve_ivp

from utils import reshape_pt1, reshape_pt1_tonormal, reshape_dim1


# Possible dynamics (take t,x,u and return dx/dt) and solver for testing

# Input x, u, version and parameters, output x at the next step (dt
# later) with scipy ODE solver
def dynamics_traj(x0, u, t0, dt, init_control, discrete=False, version=None,
                  meas_noise_var=0, process_noise_var=0, method='RK45',
                  t_span=[0, 1], t_eval=[0.1], solver_options={}, **kwargs):
    x0 = reshape_pt1(x0)
    if discrete:
        xtraj = np.zeros((len(t_eval), x0.shape[1]))
        xtraj[0] = reshape_pt1(x0)
        t = t0
        i = 0
        while (i < len(t_eval) - 1) and (t < t_eval[-1]):
            i += 1
            xnext = reshape_pt1(
                version(t, xtraj[-1], u, t0, init_control, process_noise_var,
                        **kwargs))
            xtraj[i] = xnext
            t += dt
    else:
        sol = solve_ivp(
            lambda t, x: version(t, x, u, t0, init_control, process_noise_var,
                                 **kwargs), t_span=t_span,
            y0=reshape_pt1_tonormal(x0), method=method, t_eval=t_eval,
            **solver_options)
        xtraj = reshape_pt1(sol.y.T)
    if meas_noise_var != 0:
        xtraj += np.random.normal(0, np.sqrt(meas_noise_var), xtraj.shape)
    return reshape_pt1(xtraj)


# Dynamics of the continuous time Duffing oscillator, with control law u(t)
def duffing_dynamics(t, x, u, t0, init_control, process_noise_var, kwargs):
    alpha = kwargs.get('alpha')
    beta = kwargs.get('beta')
    delta = kwargs.get('delta')
    x = reshape_pt1(x)
    u = reshape_pt1(u(t, kwargs, t0, init_control))
    A = reshape_pt1([[0, 1], [-alpha, -delta]])
    F1 = reshape_dim1(np.zeros_like(x[:, 0]))
    F2 = reshape_dim1(- beta * x[:, 0] ** 3)
    F = reshape_pt1(np.concatenate((F1, F2), axis=1))
    xdot = reshape_pt1(np.dot(A, reshape_pt1_tonormal(x)) + F + u)
    if process_noise_var != 0:
        xdot += np.random.normal(0, np.sqrt(process_noise_var), xdot.shape)
    return xdot


# Dynamics of the discrete time Duffing oscillator, with control law u(t)
def duffing_dynamics_discrete(t, x, u, t0, init_control, process_noise_var,
                              kwargs):
    alpha = kwargs.get('alpha')
    beta = kwargs.get('beta')
    delta = kwargs.get('delta')
    x = reshape_pt1(x)
    u = reshape_pt1(u(t, kwargs, t0, init_control))
    A = reshape_pt1([[0, 1], [-alpha, -delta]])
    F1 = reshape_dim1(np.zeros_like(x[:, 0]))
    F2 = reshape_dim1(- beta * x[:, 0] ** 3)
    F = reshape_pt1(np.concatenate((F1, F2), axis=1))
    xnext = reshape_pt1(np.dot(A, reshape_pt1_tonormal(x)) + F + u)
    if process_noise_var != 0:
        xnext += np.random.normal(0, np.sqrt(process_noise_var), xnext.shape)
    return xnext


# Dynamics of the continuous time Van der Pol oscillator, with control law u(t)
# See http://www.tangentex.com/VanDerPol.htm
def VanderPol_dynamics(t, x, u, t0, init_control, process_noise_var, kwargs):
    mu = kwargs.get('mu')
    x = reshape_pt1(x)
    u = reshape_pt1(u(t, kwargs, t0, init_control))
    A = reshape_pt1([[0, 1], [-1, 0]])
    F = reshape_pt1([0, mu * (1 - x[:, 0] ** 2) * x[:, 1] - x[:, 0]])
    xdot = reshape_pt1(np.dot(A, reshape_pt1_tonormal(x)) + F + u)
    if process_noise_var != 0:
        xdot += np.random.normal(0, np.sqrt(process_noise_var), xdot.shape)
    return xdot


# Dynamics of a simple inverted pendulum, with control law u(t), continuous time
# http://www.matthewpeterkelly.com/tutorials/simplePendulum/index.html
def pendulum_dynamics(t, x, u, t0, init_control, process_noise_var, kwargs):
    k = kwargs.get('k')
    m = kwargs.get('m')
    g = kwargs.get('g')
    l = kwargs.get('l')
    x = reshape_pt1(x)
    u = reshape_pt1(u(t, kwargs, t0, init_control))
    theta_before = x[:, 0]
    thetadot_before = x[:, 1]
    A = reshape_pt1([[0, 1], [0, 0]])
    F1 = reshape_dim1(np.zeros_like(x[:, 0]))
    F2 = reshape_dim1(- g / l * np.sin(theta_before) - k / m * thetadot_before)
    F = reshape_pt1(np.concatenate((F1, F2), axis=1))
    xdot = reshape_pt1(np.dot(A, reshape_pt1_tonormal(x)) + F + u)
    if process_noise_var != 0:
        xdot += np.random.normal(0, np.sqrt(process_noise_var), xdot.shape)
    return xdot


# Dynamics of a harmonic oscillator, with control law u(t), continuous time
# https://en.wikipedia.org/wiki/Harmonic_oscillator
def harmonic_oscillator_dynamics(t, x, u, t0, init_control, process_noise_var,
                                 kwargs):
    k = kwargs.get('k')
    m = kwargs.get('m')
    x = reshape_pt1(x)
    u = reshape_pt1(u(t, kwargs, t0, init_control))
    A = reshape_pt1([[0, 1], [- k / m, 0]])
    xdot = reshape_pt1(np.dot(A, reshape_pt1_tonormal(x)) + u)
    if process_noise_var != 0:
        xdot += np.random.normal(0, np.sqrt(process_noise_var), xdot.shape)
    return xdot


# Dynamics of the continuous time Duffing oscillator, with control law u(t),
# modified for nonlinearity to be of form cos(x)**2 rather than x**3
def duffing_modified_cossquare(t, x, u, t0, init_control, process_noise_var,
                               kwargs):
    alpha = kwargs.get('alpha')
    beta = kwargs.get('beta')
    delta = kwargs.get('delta')
    x = reshape_pt1(x)
    u = reshape_pt1(u(t, kwargs, t0, init_control))
    A = reshape_pt1([[0, 1], [-alpha, -delta]])
    F1 = reshape_dim1(np.zeros_like(x[:, 0]))
    F2 = reshape_dim1(- beta * (np.cos(x[:, 0]) ** 2))
    F = reshape_pt1(np.concatenate((F1, F2), axis=1))
    xdot = reshape_pt1(np.dot(A, reshape_pt1_tonormal(x)) + F + u)
    if process_noise_var != 0:
        xdot += np.random.normal(0, np.sqrt(process_noise_var), xdot.shape)
    return xdot


# Simple, gravity-lead part of the dynamics of the Wandercraft test arm
def wdc_arm_dynamics(t, x, u, t0, init_control, process_noise_var, kwargs):
    inertia = kwargs.get('inertia')
    m = kwargs.get('m')
    lG = kwargs.get('lG')
    g = kwargs.get('g')
    x = reshape_pt1(x)
    u = reshape_pt1(u(t, kwargs, t0, init_control))
    A = reshape_pt1([[0, 1], [0, 0]])
    F1 = reshape_dim1(np.zeros_like(x[:, 0]))
    F2 = reshape_dim1(-m * g * lG / inertia * np.sin(x[:, 0]))
    F = reshape_pt1(np.concatenate((F1, F2), axis=1))
    xdot = reshape_pt1(np.dot(A, reshape_pt1_tonormal(x)) + F + u)
    if process_noise_var != 0:
        xdot += np.random.normal(0, np.sqrt(process_noise_var), xdot.shape)
    return xdot


# Classic form of the mass-spring-mass system
def mass_spring_mass_dynamics_x(t, x, u, t0, init_control, process_noise_var,
                                kwargs):
    m1 = kwargs.get('m1')
    m2 = kwargs.get('m2')
    k1 = kwargs.get('k1')
    k2 = kwargs.get('k2')
    x = reshape_pt1(x)
    u = reshape_pt1(u(t, kwargs, t0, init_control))
    A = reshape_pt1([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    B = reshape_pt1([[0], [0], [0], [1]])
    F1 = reshape_dim1(np.zeros_like(x[:, 0]))
    F2 = reshape_dim1(
        k1 / m1 * (x[:, 2] - x[:, 0]) + k2 / m1 * (x[:, 2] - x[:, 0]) ** 3)
    F3 = reshape_dim1(np.zeros_like(x[:, 2]))
    F4 = reshape_dim1(
        -k1 / m2 * (x[:, 2] - x[:, 0]) - k2 / m2 * (x[:, 2] - x[:, 0]) ** 3)
    F = reshape_pt1(np.concatenate((F1, F2, F3, F4), axis=1))
    xdot = reshape_pt1(np.dot(A, reshape_pt1_tonormal(x)) + F +
                       np.dot(B, reshape_pt1_tonormal(u)))
    if process_noise_var != 0:
        xdot += np.random.normal(0, np.sqrt(process_noise_var), xdot.shape)
    return xdot


# Canonical form of the mass-spring-mass system using x1 as flat output
def mass_spring_mass_dynamics_z(t, z, u, t0, init_control, process_noise_var,
                                kwargs):
    m1 = kwargs.get('m1')
    m2 = kwargs.get('m2')
    k1 = kwargs.get('k1')
    k2 = kwargs.get('k2')
    z = reshape_pt1(z)
    z3 = reshape_pt1(z[:, 2])
    u = reshape_pt1(u(t, kwargs, t0, init_control))
    A = np.eye(z.shape[1], k=1)
    F1 = reshape_dim1(np.zeros_like(z[:, :3]))
    v = reshape_pt1_tonormal(mass_spring_mass_v(z, kwargs))
    vdot = reshape_pt1_tonormal(mass_spring_mass_vdot(z, kwargs))
    F2 = reshape_dim1(
        k1 / (m1 * m2) * (u - (m1 + m2) * z3) + (3 * k2) / (m1 * m2) * (u - (
                m1 + m2) * z3) * v ** 2 + (6 * k2) / m1 * v * vdot ** 2)
    F = reshape_pt1(np.concatenate((F1, F2), axis=1))
    zdot = reshape_pt1(np.dot(A, reshape_pt1_tonormal(z)) + F)
    if process_noise_var != 0:
        zdot += np.random.normal(0, np.sqrt(process_noise_var), zdot.shape)
    return zdot


# Utility function for the mass-spring-mass system
# Solution obtained with http://eqworld.ipmnet.ru/en/solutions/ae/ae0103.pdf
def mass_spring_mass_v(z, kwargs):
    m1 = kwargs.get('m1')
    k1 = kwargs.get('k1')
    k2 = kwargs.get('k2')
    z = reshape_pt1(z)
    x1d2 = reshape_pt1(z[:, 2])
    p = k1 / k2
    q = - m1 / k2 * x1d2
    D = np.power(p / 3, 3) + np.power(q / 2, 2)
    A = np.cbrt(-q / 2 + np.sqrt(D))
    B = np.cbrt(-q / 2 - np.sqrt(D))
    v = reshape_pt1(A + B)
    return v


# Utility function for the mass-spring-mass system
def mass_spring_mass_vdot(z, kwargs):
    m1 = kwargs.get('m1')
    k1 = kwargs.get('k1')
    k2 = kwargs.get('k2')
    z = reshape_pt1(z)
    x1d3 = reshape_pt1(z[:, 3])
    A = k1 / m1 + 3 * k2 / m1 * mass_spring_mass_v(z, kwargs) ** 2
    vdot = reshape_pt1(x1d3 / A)
    return vdot


# Flat transform (from x to z) for mass-spring-mass system
def mass_spring_mass_xtoz(x, kwargs):
    m1 = kwargs.get('m1')
    k1 = kwargs.get('k1')
    k2 = kwargs.get('k2')
    x = reshape_pt1(x)
    z = reshape_pt1(x)
    z[:, 2] = k1 / m1 * (x[:, 2] - x[:, 0]) + k2 / m1 * (x[:, 2] - x[:, 0]) ** 3
    z[:, 3] = k1 / m1 * (x[:, 3] - x[:, 1]) + \
              3 * k2 / m1 * (x[:, 3] - x[:, 1]) * (x[:, 2] - x[:, 0]) ** 2
    return reshape_pt1(z)


# Inverse transform (from z to x) for mass-spring-mass system
def mass_spring_mass_ztox(z, kwargs):
    z = reshape_pt1(z)
    x = reshape_pt1(z)
    x[:, 2] = reshape_pt1_tonormal(mass_spring_mass_v(z, kwargs)) + z[:, 0]
    x[:, 3] = reshape_pt1_tonormal(mass_spring_mass_vdot(z, kwargs)) + z[:, 1]
    return reshape_pt1(x)
